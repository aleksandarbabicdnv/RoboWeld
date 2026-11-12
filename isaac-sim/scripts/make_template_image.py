"""Make a Path A test sheet: a template image of a drawing plus the true position of its lines.

The image covers the scene's drawing area (surface_size_xy_m of the config) at --px-per-mm, black
lines on white, top edge = the +y edge of the sheet. The truth file holds the same lines in paper
metres (units "paper_m"), so a run can score how well the camera pipeline found them:

    python3 isaac-sim/scripts/make_template_image.py \
        --trajectory isaac-sim/data/act_test/Cat/69fff8521d33da91.json \
        --config isaac-sim/config/xarm7_drawing_real.yaml --out isaac-sim/outputs/templates/cat_69ff

    -> cat_69ff.png, cat_69ff_truth.json

Then, in Isaac Sim (Windows, from isaac-sim/):
    python.bat scripts\\run_two_stroke_drawing.py --config config\\xarm7_drawing_real.yaml --headless ^
        --template-image outputs\\templates\\cat_69ff.png --template-truth outputs\\templates\\cat_69ff_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from project_config import load_config  # noqa: E402
from template_strokes import render_strokes, strokes_to_trajectory  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", required=True, help="Any trajectory the loader reads (gold JSON, ...)")
    parser.add_argument("--config", default="config/xarm7_drawing_real.yaml", help="Scene config (drawing area)")
    parser.add_argument("--out", required=True, help="Output path without extension")
    parser.add_argument("--px-per-mm", type=float, default=8.0)
    parser.add_argument("--line-width-mm", type=float, default=1.0)
    parser.add_argument("--fill", type=float, default=0.9, help="Share of the drawing area the drawing may use")
    args = parser.parse_args()

    # Relative to the current directory if it exists there (repo root), else to isaac-sim/.
    config = load_config(Path(args.config).resolve() if Path(args.config).exists() else args.config, PROJECT)
    drawing = config["drawing"]
    size = np.asarray(drawing["surface_size_xy_m"], dtype=np.float64)
    trajectory = load_trajectory(args.trajectory)
    if trajectory.get("units") == "paper_m":
        lines = [np.asarray(s["points"], dtype=np.float64) for s in trajectory["strokes"]]
    else:
        mapped = map_trajectory_to_plane(
            trajectory, center_xy=[0.0, 0.0], size_xy=(size * float(args.fill)).tolist(),
            flip_image_y=bool(drawing["flip_image_y"]), max_step=0.0005,
        )
        lines = [np.asarray(s["points"], dtype=np.float64) for s in mapped["strokes"]]
    image = render_strokes(lines, size, args.px_per_mm * 1000.0, args.line_width_mm / 1000.0)

    from PIL import Image

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).convert("RGB").save(out.with_suffix(".png"))
    truth = strokes_to_trajectory(lines, f"{trajectory['drawing_id']}_truth",
                                  source_trajectory=str(Path(args.trajectory)), line_width_m=args.line_width_mm / 1000.0)
    out.with_name(out.name + "_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "image": str(out.with_suffix(".png")),
        "truth": str(out.with_name(out.name + "_truth.json")),
        "image_px": [int(image.shape[1]), int(image.shape[0])],
        "drawing_area_m": size.tolist(),
        "strokes": len(lines),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
