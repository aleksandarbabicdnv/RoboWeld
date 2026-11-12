from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from paper_surface import make_surface  # noqa: E402
from template_strokes import (  # noqa: E402
    CameraMapping,
    ScanMapping,
    compare_strokes,
    extract_strokes,
    order_strokes,
    polyline_length,
    render_strokes,
    strokes_to_trajectory,
    zhang_suen_thin,
)
from trajectory_loader import TrajectoryError, load_trajectory  # noqa: E402

SIZE = (0.16, 0.12)


def _truth():
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.c_[-0.035 + 0.025 * np.cos(t), 0.015 + 0.025 * np.sin(t)]
    bar = np.c_[np.linspace(0.0, 0.06, 100), np.full(100, 0.03)]
    stem = np.c_[np.full(80, 0.03), np.linspace(0.03, -0.02, 80)]
    wave = np.c_[np.linspace(-0.07, 0.07, 300), -0.045 + 0.006 * np.sin(np.linspace(0, 6 * np.pi, 300))]
    return [circle, bar, stem, wave]


def _photograph(template, surface, resolution=640, height=0.6, view=0.20, px_per_m=8000.0):
    """A straight-down camera image of `template` lying on `surface` (nearest-pixel texture lookup)."""
    centre = surface.point([0, 0])
    camera = CameraMapping((surface.center_xy[0], surface.center_xy[1], centre[2] + height), view, height, surface, SIZE)
    uv, _ = camera.pixel_uv((resolution, resolution))
    h, w = template.shape
    col = np.clip(((uv[..., 0] + SIZE[0] / 2) * px_per_m).astype(int), 0, w - 1)
    row = np.clip(((SIZE[1] / 2 - uv[..., 1]) * px_per_m).astype(int), 0, h - 1)
    inside = np.isfinite(uv).all(-1) & (np.abs(uv[..., 0]) <= SIZE[0] / 2) & (np.abs(uv[..., 1]) <= SIZE[1] / 2)
    image = np.full((resolution, resolution), 210, np.uint8)
    image[inside] = template[row[inside], col[inside]]
    shade = np.linspace(0.8, 1.0, resolution)[None, :]
    return (image * shade).astype(np.uint8), camera


class ThinningTests(unittest.TestCase):
    def test_thick_bar_thins_to_one_pixel(self):
        mask = np.zeros((40, 80), bool)
        mask[15:24, 10:70] = True
        skeleton = zhang_suen_thin(mask)
        columns = skeleton[:, 20:60].sum(axis=0)
        self.assertTrue((columns == 1).all())


class ExtractionTests(unittest.TestCase):
    def test_scan_round_trip(self):
        truth = _truth()
        image = render_strokes(truth, SIZE, 5000.0, 0.001)
        result = extract_strokes(image, ScanMapping(SIZE))
        report = compare_strokes(result.strokes, truth)
        self.assertEqual(len(result.strokes), 4)  # circle, T bar, T stem, wave
        self.assertLess(report["truth_to_extracted_mean_m"], 0.0002)
        self.assertEqual(report["coverage_at_1mm"], 1.0)
        closed = [s for s in result.strokes if np.linalg.norm(s[0] - s[-1]) < 1e-9]
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(polyline_length(closed[0]), 2 * np.pi * 0.025, delta=0.002)

    def test_camera_on_every_surface(self):
        truth = _truth()
        template = render_strokes(truth, SIZE, 8000.0, 0.001)
        specs = [None, {"kind": "tilted", "tilt_deg": [6, 8]},
                 {"kind": "bumps", "bumps": [{"center_uv_m": [-0.03, 0.0], "height_m": 0.008, "sigma_m": 0.02}]}]
        for spec in specs:
            surface = make_surface(spec, center_xy=(0.477, 0.088), sheet_z=0.0486, size_xy=SIZE)
            image, camera = _photograph(template, surface)
            result = extract_strokes(image, camera)
            report = compare_strokes(result.strokes, truth)
            self.assertLess(report["truth_to_extracted_mean_m"], 0.0003, spec)
            self.assertGreater(report["coverage_at_1mm"], 0.99, spec)

    def test_blank_sheet_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_strokes(np.full((300, 400), 240, np.uint8), ScanMapping(SIZE))

    def test_camera_centre_pixel_is_paper_centre(self):
        surface = make_surface(None, center_xy=(0.4, 0.1), sheet_z=0.2, size_xy=SIZE)
        camera = CameraMapping((0.4, 0.1, 0.8), 0.2, 0.6, surface, SIZE)
        uv, valid = camera.pixel_uv((100, 100))
        np.testing.assert_allclose(uv[49:51, 49:51].mean(axis=(0, 1)), [0, 0], atol=1e-9)
        self.assertGreater(uv[0, 99, 0], 0.0)   # top-right pixel: +u
        self.assertGreater(uv[0, 99, 1], 0.0)   # and +v
        self.assertFalse(valid[0, 0])           # the view is wider than the drawing area

    def test_order_reverses_to_shorten_travel(self):
        a = np.array([[0.0, 0.0], [0.01, 0.0]])
        b = np.array([[0.05, 0.0], [0.011, 0.0]])
        ordered = order_strokes([b, a], start=(0.0, 0.0))
        np.testing.assert_allclose(ordered[0], a)
        np.testing.assert_allclose(ordered[1][0], [0.011, 0.0])


class PaperUnitsTests(unittest.TestCase):
    def test_paper_metres_are_placed_unscaled(self):
        strokes = [np.array([[-0.02, 0.01], [0.03, 0.01], [0.03, -0.02]])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lines.json"
            path.write_text(json.dumps(strokes_to_trajectory(strokes, "lines")), encoding="utf-8")
            trajectory = load_trajectory(path)
        self.assertEqual(trajectory["units"], "paper_m")
        mapped = map_trajectory_to_plane(trajectory, center_xy=[0.5, 0.1], size_xy=SIZE, flip_image_y=True,
                                         max_step=0.005)
        points = np.asarray(mapped["strokes"][0]["points"])
        np.testing.assert_allclose(points[0], [0.48, 0.11])
        np.testing.assert_allclose(points[-1], [0.53, 0.08])
        self.assertEqual(mapped["mapping"]["uniform_scale"], 1.0)

    def test_lines_off_the_sheet_are_refused(self):
        trajectory = {"drawing_id": "x", "units": "paper_m", "strokes": [{"stroke_id": 0, "points": [[0, 0], [0.2, 0]]}]}
        with self.assertRaises(ValueError):
            map_trajectory_to_plane(trajectory, center_xy=[0.5, 0.1], size_xy=SIZE)

    def test_unknown_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"drawing_id": "x", "units": "inches",
                                        "strokes": [{"points": [[0, 0], [1, 1]]}]}), encoding="utf-8")
            with self.assertRaises(TrajectoryError):
                load_trajectory(path)

    def test_normalized_trajectories_are_unchanged(self):
        trajectory = load_trajectory(PROJECT_ROOT / "data" / "puppy_two_stroke.json")
        self.assertNotIn("units", trajectory)


if __name__ == "__main__":
    unittest.main()
