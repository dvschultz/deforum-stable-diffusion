"""The shipped motion presets (settings/*.json) must be valid partial settings whose
keyframe schedules parse through Deforum's DeformAnimKeys (torch-free: pandas/numexpr)."""
import ast
import os
import warnings
from types import SimpleNamespace

import pytest

from helpers.animation import DeformAnimKeys
from helpers.settings import load_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_DIR = os.path.join(ROOT, "settings")
PRESETS = sorted(f for f in os.listdir(SETTINGS_DIR) if f.endswith(".json"))


def _anim_defaults():
    """DeforumAnimArgs() defaults, ast-extracted (no full-notebook exec)."""
    src = open(os.path.join(ROOT, "Deforum_Stable_Diffusion.py")).read()
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "DeforumAnimArgs"]
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "<presets>", "exec"), ns)
    return ns["DeforumAnimArgs"]()


def test_presets_present():
    assert PRESETS, "no motion presets found in settings/"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_is_partial_motion_settings(preset):
    # A motion preset must NOT carry prompts/resolution/strength -- it sets only the move.
    import json
    jdata = json.load(open(os.path.join(SETTINGS_DIR, preset)))
    assert jdata.get("animation_mode") in ("2D", "3D")
    for forbidden in ("prompts", "W", "H", "strength", "strength_schedule", "seed"):
        assert forbidden not in jdata, f"{preset} should not override {forbidden!r}"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_schedules_parse(preset):
    anim = _anim_defaults()
    load_args({}, anim, preset, "", verbose=False)   # resolves against settings/
    a = SimpleNamespace(**anim)
    a.max_frames = 24
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # silence pandas dtype FutureWarnings
        keys = DeformAnimKeys(a)
        for i in (0, 12, 23):                        # schedules evaluate to finite numbers
            assert abs(float(keys.zoom_series[i])) < 1e6
            assert abs(float(keys.translation_x_series[i])) < 1e6
            assert abs(float(keys.angle_series[i])) < 1e6
