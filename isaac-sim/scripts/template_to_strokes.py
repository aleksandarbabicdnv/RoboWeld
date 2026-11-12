"""Path A offline: picture of line art on the sheet -> pen strokes in paper metres.

Use it on a template PNG or a flatbed scan (--mode scan: the image is exactly the drawing area),
or on a photo from a straight-down camera (--mode camera: the camera sits --camera-height-m above
the centre of the sheet and its image spans the drawing area plus --view-margin-m on each side,
as run_two_stroke_drawing.py --template-image places it; --challenge gives the sheet's shape).
The output trajectory (units "paper_m") is drawn where the lines are by both runners:

    python3 isaac-sim/scripts/template_to_strokes.py --image outputs/templates/cat_69ff.png --mode scan \
        --config isaac-sim/config/xarm7_drawing_real.yaml --out outputs/templates/cat_69ff_strokes.json \
        --truth outputs/templates/cat_69ff_truth.json --preview outputs/templates/cat_69ff_strokes.png

    # then draw it: run_two_stroke_drawing.py --trajectory ...cat_69ff_strokes.json
    # or the MCP server: draw_trajectory(trajectory_path=".../cat_69ff_strokes.json")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paper_surface import make_surface  # noqa: E402
from project_config import load_config  # noqa: E402
from template_strokes import (  # noqa: E402
    CameraMapping,
    ScanMapping,
    compare_strokes,
    extract_strokes,
    polyline_length,
    strokes_to_trajectory,
)
from trajectory_loader import load_trajectory  # noqa: E402


def _preview(path: Path, image: np.ndarray, strokes, mapping, size) -> None:
    """Extracted strokes drawn in colour over a scan-style view of the drawing area."""
    from PIL import Image, ImageDraw

    scale = 6000.0  # px per metre
    width, height = int(size[0] * scale), int(size[1] * scale)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    colours = [(220, 30, 30), (30, 120, 220), (20, 160, 60), (200, 120, 0), (140, 40, 180), (0, 150, 150)]
    for k, s in enumerate(strokes):
        pts = [((u + size[0] / 2) * scale, (size[1] / 2 - v) * scale) for u, v in s]
        draw.line(pts, fill=colours[k % len(colours)], width=3)
        draw.ellipse([pts[0][0] - 5, pts[0][1] - 5, pts[0][0] + 5, pts[0][1] + 5], outline=(0, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config/xarm7_drawing_real.yaml")
    parser.add_argument("--mode", choices=["scan", "camera"], default="scan")
    parser.add_argument("--challenge", default=None, help="camera mode: sheet shape from this challenge preset")
    parser.add_argument("--camera-height-m", type=float, default=None, help="default: template_camera in the presets")
    parser.add_argument("--view-margin-m", type=float, default=None)
    parser.add_argument("--out", required=True, help="Trajectory JSON to write")
    parser.add_argument("--truth", default=None, help="paper_m trajectory of the true lines, to score against")
    parser.add_argument("--preview", default=None, help="PNG with the extracted strokes, one colour per stroke")
    args = parser.parse_args()

    from PIL import Image

    # Relative to the current directory if it exists there (repo root), else to isaac-sim/.
    config = load_config(Path(args.config).resolve() if Path(args.config).exists() else args.config, PROJECT)
    drawing = config["drawing"]
    size = [float(v) for v in drawing["surface_size_xy_m"]]
    image = np.asarray(Image.open(args.image).convert("RGB"))
    if args.mode == "scan":
        mapping = ScanMapping(tuple(size))
    else:
        from surface_challenge import Challenge, load_presets

        presets = load_presets()
        camera_cfg = presets.get("template_camera") or {}
        if args.challenge:
            surface = Challenge.load(args.challenge, drawing).surface
        else:
            surface = make_surface(None, center_xy=drawing["surface_center_xy_m"], sheet_z=float(drawing["paper_top_z_m"]),
                                   size_xy=size)
        height = args.camera_height_m if args.camera_height_m is not None else float(camera_cfg.get("height_above_sheet_m", 0.6))
        margin = args.view_margin_m if args.view_margin_m is not None else float(camera_cfg.get("view_margin_m", 0.02))
        centre = surface.point([0.0, 0.0])
        eye = (float(surface.center_xy[0]), float(surface.center_xy[1]), float(centre[2] + height))
        mapping = CameraMapping(eye, max(size) + 2 * margin, height, surface, tuple(size),
                                border_m=float(camera_cfg.get("border_m", 0.002)))
    result = extract_strokes(image, mapping)
    trajectory = strokes_to_trajectory(result.strokes, Path(args.image).stem, source_image=str(args.image),
                                       mode=args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trajectory, indent=1), encoding="utf-8")
    report = {
        "status": "ok" if result.strokes else "no_lines",
        "out": str(out),
        "strokes": len(result.strokes),
        "stroke_length_m": float(sum(polyline_length(s) for s in result.strokes)),
        "pixel_size_m": result.pixel_size_m,
        "threshold": result.threshold,
    }
    if args.truth:
        truth = load_trajectory(args.truth)
        report["vs_truth"] = compare_strokes(result.strokes, [np.asarray(s["points"]) for s in truth["strokes"]])
    if args.preview:
        _preview(Path(args.preview), image, result.strokes, mapping, size)
        report["preview"] = args.preview
    print(json.dumps(report, indent=2))
    return 0 if result.strokes else 2


if __name__ == "__main__":
    raise SystemExit(main())
