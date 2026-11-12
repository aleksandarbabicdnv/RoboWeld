# Surface challenges and camera templates (Path A)

Two additions to the headless drawing runner (`scripts/run_two_stroke_drawing.py`):

1. **Surface challenges** (`--challenge NAME`): draw on a tilted or bumpy sheet, with the pen
   leaned at an angle, and with a pen-force target through a spring-loaded pen model.
2. **Camera templates** (`--template-image PNG`): put line art on the sheet, photograph it with a
   straight-down camera, turn the photo into pen strokes and draw them where the lines are.

Neither changes the planner, the controllers' tuning or the calibration configs. A challenge only
moves the existing motion targets onto the surface and gives each one a pen orientation and force
target. Without the new options the runner behaves exactly as before.

## Quick start (Windows Command Prompt, in `isaac-sim\`)

```bat
set "ISAAC_ROOT=C:\isaac-sim-6.0.1"
set "PYTHONEXE=%ISAAC_ROOT%\kit\python\python.exe"

rem Sanity check: must match a plain run
call "%ISAAC_ROOT%\python.bat" scripts\run_two_stroke_drawing.py --config config\xarm7_drawing_real.yaml ^
    --trajectory data\act_test\Cat\69fff8521d33da91.json --challenge flat --headless --output-dir outputs\challenges\flat

rem Level 2: tilted sheet
call "%ISAAC_ROOT%\python.bat" scripts\run_two_stroke_drawing.py --config config\xarm7_drawing_real.yaml ^
    --trajectory data\act_test\Cat\69fff8521d33da91.json --challenge tilted_8_6 --headless --output-dir outputs\challenges\tilted_8_6
```

Omit `--headless` to watch. Results land in `--output-dir`: `run_metrics.json` (new section
`surface_challenge`), `executed_path.csv` (new per-sample columns), and the usual plots.

Path A, end to end:

```bat
rem 1. Make a test sheet from any drawing (any Python with numpy, Pillow and PyYAML)
call "%ISAAC_ROOT%\python.bat" scripts\make_template_image.py --trajectory data\act_test\Cat\69fff8521d33da91.json ^
    --config config\xarm7_drawing_real.yaml --out outputs\templates\cat_69ff

rem 2. Photograph it in Isaac Sim, extract the strokes, draw them (any challenge works too)
call "%ISAAC_ROOT%\python.bat" scripts\run_two_stroke_drawing.py --config config\xarm7_drawing_real.yaml ^
    --template-image outputs\templates\cat_69ff.png --template-truth outputs\templates\cat_69ff_truth.json ^
    --headless --output-dir outputs\challenges\template_cat
```

The run writes `template_camera.png` (what the camera saw) and `template_strokes.json` (the strokes
it found, units `paper_m`), and `run_metrics.json` gets a `template` section with the extraction
score against the truth. `template_strokes.json` can also be drawn by the MCP flow:
`draw_trajectory(trajectory_path=".../template_strokes.json")`.

Offline, without Isaac Sim (a scan, a phone photo of a printout, or the template PNG itself):

```bash
python3 isaac-sim/scripts/template_to_strokes.py --image outputs/templates/cat_69ff.png --mode scan \
    --config isaac-sim/config/xarm7_drawing_real.yaml --out outputs/templates/cat_69ff_strokes.json \
    --truth outputs/templates/cat_69ff_truth.json --preview outputs/templates/cat_69ff_strokes.png
```

## Challenge presets (`config/surface_challenges.yaml`)

| Level | Preset | What changes |
|---|---|---|
| 0 | `flat` | Nothing; reproduces a plain run (sanity check) |
| 1 | `pen_lead_15` | Pen top trails the direction of travel by 15 deg |
| 1 | `pen_fixed_20` | Pen leaned a fixed 20 deg toward the robot base |
| 2 | `tilted_8`, `tilted_8_6` | Sheet tilted 8 deg, or 6 and 8 deg about x and y (about 10 deg) |
| 2 | `tilted_8_6_lead_10` | Tilted sheet and a 10 deg lead angle |
| 3 | `bumps_3` | Three Gaussian bumps, up to 6 mm high, slopes up to about 14 deg |
| 4 | `spring_constant` | Spring-loaded pen, hold 1.5 N |
| 5 | `spring_sine` | Force follows a sine, 1.5 +- 0.7 N, 30 mm period along the stroke |
| all | `tilted_lead_spring_ramp` | Tilted sheet, 10 deg lead, force ramping 0.8 to 2.2 N |

Add your own preset by copying one. A tilted sheet is lifted so its lowest corner stays at the
original paper height (as if a wedge were slid under it), so it never goes into the table.

## How it works

- `src/paper_surface.py`: sheet shapes (flat, tilted plane, Gaussian bumps). Each maps paper
  coordinates (u, v), metres from the centre of the drawing area, to world points and normals,
  measures a point's height above the sheet, and intersects camera rays with it. Also the pen
  orientation maths: the pen axis is the IK frame's +Z; the flat default `[0, 1, 0, 0]` is
  "axis = -normal, frame x = world x".
- `src/surface_challenge.py`: `Challenge.apply()` takes the normal plan from
  `build_motion_sequence()` and moves every target onto the surface, keeping its height above the
  sheet along the local normal. Drawing targets get an orientation (normal plus lead/side angle;
  the travel direction is smoothed over 6 mm and the lead fades out at hairpins so the wrist never
  flips) and a force target. Pen-up, lowering and travel targets blend orientation smoothly between
  strokes. `measure()` and `summary()` turn each pen-down sample into the challenge metrics.
- `src/pen_spring.py`: the spring-loaded holder. The arm is stiff and position controlled, so the
  force is preload + stiffness x compression, where compression is how far the uncompressed tip is
  pushed past the touching height. A force target becomes a commanded compression. Ink is laid
  above `ink_threshold_n`; its width grows with force. Past the holder's travel the pen bottoms out
  (rigid contact), which is counted as an over-press. With a spring the sheet has no collision, so
  the model, not a rigid contact, sets the force.
- `src/template_strokes.py`: dark lines -> threshold (after dividing out slow shading) -> one-pixel
  centre lines -> paths between ends and junctions -> branches that continue straight through a
  junction are joined (a T gives the bar plus the stem, an X gives two lines, a closed outline
  stays one loop) -> smoothed, simplified, ordered for short pen-up travel. Pure numpy, so it runs
  in Isaac Sim's Python and in WSL.
- `src/surface_scene.py`: the Isaac side: the sheet as a triangle mesh (with a static collider when
  the sheet should collide), the template as a textured overlay that follows the surface, the
  straight-down camera and the photo (Replicator `rgb` annotator, same as `video_capture_node.py`).
  The photo is taken before the robot is added, so the arm can never hide the sheet.
- Trajectories with `"units": "paper_m"` (what the template pipeline writes) are placed as they
  are instead of being scaled to fit the drawing area (`src/coordinate_mapper.py`), and the loader
  keeps that field. Every other trajectory is unchanged.

## Metrics (`run_metrics.json` -> `surface_challenge`)

| Key | Meaning |
|---|---|
| `surface_path_error` | Distance in paper coordinates between the planned pen-tip path and the drawn one, both ways; `inked_to_desired` only counts samples that left ink |
| `ink.coverage_fraction`, `ink.gap_length_m`, `ink.longest_gap_m` | Share of the planned path with ink within 1 mm, and the length of what was missed |
| `height_above_contact_m` | Pen tip height above its touching height, along the normal (negative = pressed in) |
| `tilt_error_deg` | Angle between the commanded and the actual pen axis |
| `pen_force_n` | Spring runs: force statistics, `target_error_rmse_n`, `in_band_fraction` (within 0.5 N), `contact_fraction`, `bottomed_out_samples` |

`executed_path.csv` gets the per-sample columns behind these numbers (`pen_tip_*`, `paper_u_m`,
`paper_v_m`, `height_above_contact_m`, `compression_m`, `pen_force_n`, `target_force_n`, `inked`,
`bottomed_out`, `tilt_error_deg`).

## Limits of this first version

- Controllers: `ik` and `ik_stream` only. The trained ACT and SmolVLA policies know only the flat
  sheet and a vertical pen; the runner refuses a challenge with `--controller act`. Retraining
  needs the kinematic labeller (`src/kinematic_labeller.py`) to label challenge episodes.
- The spring pen is a model computed from the measured tip pose, not a PhysX prismatic joint with
  a contact sensor. That is accurate for a stiff position-controlled arm; a physical joint is the
  next step if contact dynamics (bouncing, friction) matter.
- The MCP Action Graph controller (`scripts/mcp_drawing_controller.py`) does not run challenges
  yet. It does draw `paper_m` template trajectories, because those only need the mapper change.
- The live ink in the MCP video capture still assumes a flat sheet.
- Untested in Isaac Sim at the time of writing; all pure-Python parts are unit tested
  (`tests/test_surface_challenge.py`, `tests/test_template_strokes.py`).
