# Motion presets

Drop-in camera-motion presets for 2D/3D animation. Each file is a **partial** settings
JSON: `helpers/settings.load_args` overrides only the keys present, so a preset sets the
*motion* (and `animation_mode`) and leaves everything else — prompts, resolution, model,
strength, steps — at your in-notebook / in-script values.

The schedules are **length-agnostic** (constant rates or `sin/cos` of the frame index `t`),
so they work at any `max_frames`.

## Use it

**Notebook** (Load Settings cell):
- `override_settings_with_file = True`
- `settings_file = "2d_slow_drift.json"` (pick from the dropdown)

**Script / programmatic:**
```python
from helpers.settings import load_args
load_args(args_dict, anim_args_dict, "2d_orbit.json", custom_settings_file="", verbose=False)
```
(`settings_file` is resolved against this `settings/` dir; use `"custom"` +
`custom_settings_file=/abs/path.json` for one elsewhere.)

## Presets

| File | Move |
|---|---|
| `2d_slow_drift.json` | gentle breathing zoom + slow rotate + sine sway (the films' look) |
| `2d_push_in.json` | steady zoom toward center (dolly in) |
| `2d_pull_back.json` | steady zoom out (reveal) |
| `2d_orbit.json` | continuous rotation + slight zoom |
| `2d_sway.json` | horizontal/vertical oscillation, no zoom |
| `2d_spiral.json` | zoom in + continuous rotation |
| `2d_handheld.json` | small irregular jitter (handheld feel) |
| `3d_push_through.json` | fly forward through the scene (`translation_z`) |
| `3d_camera_pan.json` | look left↔right pan + slow forward |

**3D presets need depth warping** — install with `python install_requirements.py --with-3d`
(MiDaS) and they only apply in `animation_mode="3D"`.

## Motion vocabulary (roll your own)

Values are Deforum keyframe schedules: `"frame:(expr), frame:(expr)"`, where `expr` is a
math expression in `t` (the frame index). Keys:

- `zoom` — per-frame scale (`1.0` none, `>1` in, `<1` out)
- `angle` — 2D rotation, degrees/frame
- `translation_x` / `translation_y` — pixels/frame
- `translation_z` — forward/back (3D only), `rotation_3d_x/y/z` — pitch/yaw/roll (3D only)

e.g. `"0:(3*sin(2*3.14*t/120))"` sways ±3px with a 120-frame period; `"0:(1.0), 180:(1.04)"`
ramps zoom from frame 0 to 180. Tip: a render also writes its full config to
`{timestring}_settings.txt` in the output dir — copy/trim that into a new preset.
