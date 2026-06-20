"""Legacy settings files (with now-removed SD keys) must still load cleanly.

load_args only copies keys that exist in the target args dict, so a saved
settings file carrying `sampler`, `scale`, `clip_scale`, etc. loads without error
and those keys never appear on the new args namespace.
"""
import json

from helpers.settings import load_args


def test_load_args_tolerates_legacy_sd_keys(tmp_path):
    # New-world args dict (post-prune): no sampler/scale/clip_scale.
    args_dict = {"W": 512, "H": 512, "steps": 8, "strength": 0.65, "acceleration": "regular"}
    anim_args_dict = {"animation_mode": "2D", "max_frames": 100}

    # A legacy settings file mixes known keys with removed SD-only keys.
    legacy = {
        "W": 768, "H": 768, "steps": 50,          # known: should be copied
        "sampler": "euler_ancestral",              # removed: must be ignored
        "scale": 7, "clip_scale": 0.5,             # removed: must be ignored
        "sampler_schedule": "0:('euler')",         # removed: must be ignored
        "animation_mode": "3D",                    # known anim key
    }
    settings_path = tmp_path / "legacy_settings.txt"
    settings_path.write_text(json.dumps(legacy))

    load_args(args_dict, anim_args_dict, "custom", str(settings_path), verbose=False)

    # Known keys updated from the file
    assert args_dict["W"] == 768 and args_dict["H"] == 768
    assert args_dict["steps"] == 50  # client clamps this to 8 at call time
    assert anim_args_dict["animation_mode"] == "3D"

    # Removed SD keys did not leak onto the args dict
    for k in ("sampler", "scale", "clip_scale", "sampler_schedule"):
        assert k not in args_dict

    # acceleration kept its default (legacy file had no such key)
    assert args_dict["acceleration"] == "regular"


def test_load_args_missing_file_is_noop(tmp_path):
    args_dict = {"W": 512}
    load_args(args_dict, {}, "custom", str(tmp_path / "does_not_exist.txt"), verbose=False)
    assert args_dict["W"] == 512  # unchanged, no exception
