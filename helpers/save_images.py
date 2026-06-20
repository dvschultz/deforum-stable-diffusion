from __future__ import annotations  # defer annotation eval so torch isn't needed at import
from typing import List, Tuple
from einops import rearrange
import numpy as np, os
from PIL import Image
import time

# torch / torchvision are imported lazily inside save_samples so get_output_folder
# (the only export used by the live path) stays torch-free.


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def save_samples(
    args, x_samples: torch.Tensor, seed: int, n_rows: int
) -> Tuple[Image.Image, List[Image.Image]]:
    """Function to save samples to disk.
    Args:
        args: Stable deforum diffusion arguments.
        x_samples: Samples to save.
        seed: Seed for the experiment.
        n_rows: Number of rows in the grid.
    Returns:
        A tuple of the grid image and a list of the generated images.
        ( grid_image, generated_images )
    """

    import torch
    from torchvision.utils import make_grid

    # save samples
    images = []
    grid_image = None
    if args.display_samples or args.save_samples:
        for index, x_sample in enumerate(x_samples):
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            images.append(Image.fromarray(x_sample.astype(np.uint8)))
            if args.save_samples:
                images[-1].save(
                    os.path.join(
                        args.outdir, f"{args.timestring}_{index:02}_{seed}.png"
                    )
                )

    # save grid
    if args.display_grid or args.save_grid:
        grid = torch.stack([x_samples], 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=n_rows, padding=0)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        grid_image = Image.fromarray(grid.astype(np.uint8))
        if args.save_grid:
            grid_image.save(
                os.path.join(args.outdir, f"{args.timestring}_{seed}_grid.png")
            )

    # return grid_image and individual sample images
    return grid_image, images
