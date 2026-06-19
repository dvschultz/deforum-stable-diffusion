import os
import shutil
import subprocess
import time
import gc
import sys

import torch
import random
from collections import OrderedDict
from types import SimpleNamespace
from cog import BasePredictor, Input, Path

sys.path.insert(0, "src")

from helpers.render import (
    render_animation,
    render_input_video,
    render_image_batch,
    render_interpolation,
)
from helpers.prompts import Prompts
from helpers.zimage_client import resolve_fal_key


class Predictor(BasePredictor):
    def setup(self):
        """Validate fal.ai auth once. Generation runs on the hosted Z-Image Turbo
        model, so there are no local weights to load."""
        resolve_fal_key()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(
        self,
        max_frames: int = Input(
            description="Number of frames for animation", default=200
        ),
        animation_prompts: str = Input(
            default="0: a beautiful apple, trending on Artstation | 50: a beautiful banana, trending on Artstation | 100: a beautiful coconut, trending on Artstation | 150: a beautiful durian, trending on Artstation",
            description="Prompt for animation. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the max_frames.",
        ),
        negative_prompts: str = Input(
            default="0: mountain",
            description="Prompt for negative. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the max_frames.",
        ),
        width: int = Input(
            description="Width of output video. Reduce if out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Reduce if out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps (Z-Image Turbo: 1-8)", ge=1, le=8, default=8
        ),
        acceleration: str = Input(
            description="Z-Image Turbo acceleration", default="regular",
            choices=["none", "regular", "high"],
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        fps: int = Input(
            default=15, ge=10, le=60, description="Choose fps for the video."
        ),
        use_init: bool = Input(
            default=False,
            description="If not using init image, you can skip the next settings to setting the animation_mode.",
        ),
        init_image: Path = Input(
            default=None, description="Provide init_image if use_init"
        ),
        strength: float = Input(
            default=0.5,
            description="The initial diffusion on the input image."
        ), 
        use_mask: bool = Input(default=False),
        mask_file: Path = Input(
            default=None, description="Provide mask_file if use_mask"
        ),
        invert_mask: bool = Input(default=False),
        animation_mode: str = Input(
            default="2D",
            choices=["2D", "3D", "Video Input", "Interpolation"],
            description="Choose Animation mode. All parameters below are for setting up animations.",
        ),
        border: str = Input(default="replicate", choices=["wrap", "replicate"]),
        angle: str = Input(
            description="angle parameter for the motion", default="0:(0)"
        ),
        zoom: str = Input(
            description="zoom parameter for the motion", default="0:(1.04)"
        ),
        translation_x: str = Input(
            description="translation_x parameter for the 2D motion",
            default="0:(10*sin(2*3.14*t/10))",
        ),
        translation_y: str = Input(
            description="translation_y parameter for the 2D motion", default="0:(0)"
        ),
        translation_z: str = Input(
            description="translation_z parameter for the 2D motion", default="0:(10)"
        ),
        rotation_3d_x: str = Input(
            description="rotation_3d_x parameter for the 3D motion", default="0:(0)"
        ),
        rotation_3d_y: str = Input(
            description="rotation_3d_y parameter for the 3D motion", default="0:(0)"
        ),
        rotation_3d_z: str = Input(
            description="rotation_3d_z parameter for the 3D motion", default="0:(0)"
        ),
        flip_2d_perspective: bool = Input(default=False),
        perspective_flip_theta: str = Input(default="0:(0)"),
        perspective_flip_phi: str = Input(default="0:(t%15)"),
        perspective_flip_gamma: str = Input(default="0:(0)"),
        perspective_flip_fv: str = Input(default="0:(53)"),
        noise_schedule: str = Input(default="0: (0.02)"),
        strength_schedule: str = Input(default="0: (0.65)"),
        contrast_schedule: str = Input(default="0: (1.0)"),
        hybrid_video_comp_alpha_schedule: str = Input(default="0:(1)"),
        hybrid_video_comp_mask_blend_alpha_schedule: str = Input(default="0:(0.5)"),
        hybrid_video_comp_mask_contrast_schedule: str = Input(default="0:(1)"),
        hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule: str = Input(
            default="0:(100)"
        ),
        hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule: str = Input(
            default="0:(0)"
        ),

        kernel_schedule: str = Input(default="0: (5)"),
        sigma_schedule: str = Input(default="0: (1.0)"),
        amount_schedule: str = Input(default="0: (0.2)"),
        threshold_schedule: str = Input(default="0: (0.0)"),
        color_coherence: str = Input(
            choices=[
                "Match Frame 0 HSV",
                "Match Frame 0 LAB",
                "Match Frame 0 RGB",
                "Video Input",
            ],
            default="Match Frame 0 LAB",
        ),
        color_coherence_video_every_N_frames: int = Input(default=1),
        color_force_grayscale: bool = Input(default=False),
        diffusion_cadence: str = Input(
            choices=["1", "2", "3", "4", "5", "6", "7", "8"],
            default="1",
        ),
        use_depth_warping: bool = Input(default=True),
        midas_weight: float = Input(default=0.3),
        near_plane: int = Input(default=200),
        far_plane: int = Input(default=10000),
        fov: int = Input(default=40),
        padding_mode: str = Input(
            choices=["border", "reflection", "zeros"],
            default="border",
        ),
        sampling_mode: str = Input(
            choices=["bicubic", "bilinear", "nearest"],
            default="bicubic",
        ),
        video_init_path: Path = Input(default=None),
        extract_nth_frame: int = Input(default=1),
        overwrite_extracted_frames: bool = Input(default=True),
        use_mask_video: bool = Input(default=False),
        video_mask_path: Path = Input(default=None),
        hybrid_video_generate_inputframes: bool = Input(default=False),
        hybrid_video_use_first_frame_as_init_image: bool = Input(default=True),
        hybrid_video_motion: str = Input(
            choices=["None", "Optical Flow", "Perspective", "Affine"],
            default="None",
        ),
        hybrid_video_flow_method: str = Input(
            choices=["Farneback", "DenseRLOF", "SF"],
            default="Farneback",
        ),
        hybrid_video_composite: bool = Input(default=False),
        hybrid_video_comp_mask_type: str = Input(
            choices=["None", "Depth", "Video Depth", "Blend", "Difference"],
            default="None",
        ),
        hybrid_video_comp_mask_inverse: bool = Input(default=False),
        hybrid_video_comp_mask_equalize: str = Input(
            choices=["None", "Before", "After", "Both"],
            default="None",
        ),
        hybrid_video_comp_mask_auto_contrast: bool = Input(default=False),
        hybrid_video_comp_save_extra_frames: bool = Input(default=False),
        hybrid_video_use_video_as_mse_image: bool = Input(default=False),
        interpolate_key_frames: bool = Input(default=False),
        interpolate_x_frames: int = Input(default=4),
        resume_from_timestring: bool = Input(default=False),
        resume_timestring: str = Input(default=""),
    ) -> Path:
        """Run a single prediction on the model"""

        # sanity checks:
        if use_init:
            assert init_image, "Please provide init_image when use_init is set to True."
        if use_mask:
            assert mask_file, "Please provide mask_file when use_mask is set to True."

        animation_prompts_dict = {}
        animation_prompts = animation_prompts.split("|")
        assert len(animation_prompts) > 0, "Please provide valid prompt for animation."
        if len(animation_prompts) == 1:
            animation_prompts = {0: animation_prompts[0]}
        else:
            for frame_prompt in animation_prompts:
                frame_prompt = frame_prompt.split(":")
                assert (
                    len(frame_prompt) == 2
                ), "Please follow the 'frame_num: prompt' format."
                frame_id, prompt = frame_prompt[0].strip(), frame_prompt[1].strip()
                assert (
                    frame_id.isdigit() and 0 <= int(frame_id) <= max_frames
                ), "frame_num should be an integer and 0<= frame_num <= max_frames"
                assert (
                    int(frame_id) not in animation_prompts_dict
                ), f"Duplicate prompts for frame_num {frame_id}. "
                assert len(prompt) > 0, "prompt cannot be empty"
                animation_prompts_dict[int(frame_id)] = prompt
            animation_prompts = OrderedDict(sorted(animation_prompts_dict.items()))

        root = {"device": self.device, "models_path": "models", "configs_path": "configs"}
        # No local weights: root.model is a lightweight Z-Image Turbo handle.
        root["model"] = SimpleNamespace(backend="z-image-turbo")
        root = SimpleNamespace(**root)

        # using some of the default settings for simplicity
        args_dict = {
            "W": width,
            "H": height,
            "bit_depth_output": 8,
            "seed": seed,
            "steps": num_inference_steps,
            "acceleration": acceleration,
            "save_samples": False,
            "save_settings": False,
            "display_samples": False,
            "n_batch": 1,
            "batch_name": "StableFun",
            "filename_format": "{timestring}_{index}_{prompt}.png",
            "seed_behavior": "iter",
            "seed_iter_N": 1,
            "make_grid": False,
            "grid_rows": 2,
            "outdir": "cog_temp_output",
            "use_init": use_init,
            "strength": strength,
            "strength_0_no_init": True,
            "init_image": init_image,
            "use_mask": use_mask,
            "use_alpha_as_mask": False,
            "mask_file": mask_file,
            "invert_mask": invert_mask,
            "mask_brightness_adjust": 1.0,
            "mask_contrast_adjust": 1.0,
            "overlay_mask": True,
            "mask_overlay_blur": 5,
            "n_samples": 1,
            "prompt": "",
            "timestring": "",
            "init_sample": None,
            "init_sample_raw": None,
            "mask_sample": None,
            "init_c": None,
            "seed_internal": 0,
        }

        anim_args_dict = {
            # Animation
            "animation_mode": animation_mode,
            "max_frames": max_frames,
            "border": border,

            #Motion Parameters
            "angle": angle,
            "zoom": zoom,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_z": translation_z,
            "rotation_3d_x": rotation_3d_x,
            "rotation_3d_y": rotation_3d_y,
            "rotation_3d_z": rotation_3d_z,
            "flip_2d_perspective": flip_2d_perspective,
            "perspective_flip_theta": perspective_flip_theta,
            "perspective_flip_phi": perspective_flip_phi,
            "perspective_flip_gamma": perspective_flip_gamma,
            "perspective_flip_fv": perspective_flip_fv,
            "noise_schedule": noise_schedule,
            "strength_schedule": strength_schedule,
            "contrast_schedule": contrast_schedule,
            "hybrid_comp_alpha_schedule": hybrid_video_comp_alpha_schedule,
            "hybrid_comp_mask_blend_alpha_schedule": hybrid_video_comp_mask_blend_alpha_schedule,
            "hybrid_comp_mask_contrast_schedule": hybrid_video_comp_mask_contrast_schedule,
            "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule,
            "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule,

            # Unsharp mask (anti-blur) Parmaters
            "kernel_schedule": kernel_schedule,
            "sigma_schedule": sigma_schedule,
            "amount_schedule": amount_schedule,
            "threshold_schedule": threshold_schedule,
            
            # Coherence
            "color_coherence": color_coherence,
            "color_coherence_video_every_N_frames": color_coherence_video_every_N_frames,
            "color_force_grayscale": color_force_grayscale,
            "diffusion_cadence": diffusion_cadence,
            
            # 3D Depth Waping
            "use_depth_warping": use_depth_warping,
            "midas_weight": midas_weight,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "fov": fov,
            "padding_mode": padding_mode,
            "sampling_mode": sampling_mode,
            "save_depth_maps": False,
            
            # Video Input
            "video_init_path": str(video_init_path),
            "extract_nth_frame": extract_nth_frame,
            "overwrite_extracted_frames": overwrite_extracted_frames,
            "use_mask_video": use_mask_video,
            "video_mask_path": str(video_mask_path),

            # Hybrid Video for 2D/3D Animation Mode
            "hybrid_generate_inputframes": hybrid_video_generate_inputframes,
            "hybrid_use_first_frame_as_init_image": hybrid_video_use_first_frame_as_init_image,
            "hybrid_motion": hybrid_video_motion,
            "hybrid_flow_method": hybrid_video_flow_method,
            "hybrid_composite": hybrid_video_composite,
            "hybrid_comp_mask_type": hybrid_video_comp_mask_type,
            "hybrid_comp_mask_inverse": hybrid_video_comp_mask_inverse,
            "hybrid_comp_mask_equalize": hybrid_video_comp_mask_equalize,
            "hybrid_comp_mask_auto_contrast": hybrid_video_comp_mask_auto_contrast,
            "hybrid_comp_save_extra_frames": hybrid_video_comp_save_extra_frames,
            "hybrid_use_video_as_mse_image": hybrid_video_use_video_as_mse_image,

            # Interpolation
            "interpolate_key_frames": interpolate_key_frames,
            "interpolate_x_frames": interpolate_x_frames,

            # Resume Animation
            "resume_from_timestring": resume_from_timestring,
            "resume_timestring": resume_timestring,            
        }

        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)

        if os.path.exists(args.outdir):
            shutil.rmtree(args.outdir)
        os.makedirs(args.outdir, exist_ok=True)

        args.timestring = time.strftime("%Y%m%d%H%M%S")
        args.strength = max(0.0, min(1.0, args.strength))

        if args.seed is None:
            args.seed = random.randint(0, 2**32 - 1)
        if not args.use_init:
            args.init_image = None

        if anim_args.animation_mode == "None":
            anim_args.max_frames = 1
        elif anim_args.animation_mode == "Video Input":
            args.use_init = True

        # clean up unused memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # get prompts
        cond, uncond = Prompts(prompt=animation_prompts,neg_prompt=negative_prompts).as_dict()

        # dispatch to appropriate renderer
        if anim_args.animation_mode == "2D" or anim_args.animation_mode == "3D":            
            render_animation(root, anim_args, args, cond, uncond)
        elif anim_args.animation_mode == "Video Input":            
            render_input_video(root, anim_args, args, cond, uncond)
        elif anim_args.animation_mode == "Interpolation":            
            render_interpolation(root, anim_args, args, cond, uncond)
        else:
            render_image_batch(root, args, cond, uncond)

        # make video
        image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
        mp4_path = f"/tmp/out.mp4"

        # make video
        cmd = [
            "ffmpeg",
            "-y",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-start_number",
            str(0),
            "-i",
            image_path,
            "-frames:v",
            str(anim_args.max_frames),
            "-c:v",
            "libx264",
            "-vf",
            f"fps={fps}",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            mp4_path,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

        return Path(mp4_path)
