from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from drawing_state_machine import MotionTarget, build_motion_sequence, flatten_desired_targets  # noqa: E402
from paper_surface import (  # noqa: E402
    Bumps,
    TiltedPlane,
    angle_between,
    make_surface,
    matrix_from_quat,
    orientation_from_axis,
    pen_axis,
    quat_from_matrix,
    slerp,
    smoothed_tangents,
    tilted_pen_axis,
)
from pen_spring import SpringPen, force_profile  # noqa: E402
from project_config import load_config  # noqa: E402
from surface_challenge import Challenge, ChallengeError, load_presets  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402

CONFIGS = ["config/xarm7_drawing.yaml", "config/xarm7_drawing_real.yaml"]


def _plan(config_name, trajectory="data/puppy_two_stroke.json"):
    config = load_config(config_name, PROJECT_ROOT)
    d = config["drawing"]
    mapped = map_trajectory_to_plane(
        load_trajectory(PROJECT_ROOT / trajectory), center_xy=d["surface_center_xy_m"], size_xy=d["surface_size_xy_m"],
        flip_image_y=d["flip_image_y"], max_step=d["max_cartesian_step_m"], smoothing_strength=d["smoothing_strength"],
        corner_angle_degrees=d["corner_angle_degrees"],
    )
    phases = build_motion_sequence(
        mapped["strokes"], pen_down_z=d["pen_down_z_m"], pen_up_z=d["pen_up_z_m"], approach_height=d["approach_height_m"],
        max_cartesian_step=d["max_cartesian_step_m"], corner_angle_degrees=d["corner_angle_degrees"],
        corner_densify_window=d["corner_window_points"], corner_densify_factor=d["corner_densify_factor"],
        short_hop_m=d.get("short_hop_m", 0.0),
    )
    return config, phases


class RotationTests(unittest.TestCase):
    def test_vertical_pen_gives_configured_orientation(self):
        np.testing.assert_allclose(orientation_from_axis([0, 0, -1]), [0, 1, 0, 0], atol=1e-12)

    def test_quaternion_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            q = -q if q[0] < 0 else q
            np.testing.assert_allclose(quat_from_matrix(matrix_from_quat(q)), q, atol=1e-9)

    def test_orientation_axis_is_requested_axis(self):
        axis = np.array([0.3, -0.2, -0.9])
        np.testing.assert_allclose(pen_axis(orientation_from_axis(axis)), axis / np.linalg.norm(axis), atol=1e-9)

    def test_slerp_ends_and_middle(self):
        a, b = orientation_from_axis([0, 0, -1]), orientation_from_axis([0.5, 0, -0.866])
        np.testing.assert_allclose(slerp(a, b, 0.0), a, atol=1e-9)
        np.testing.assert_allclose(slerp(a, b, 1.0), b, atol=1e-9)
        mid = angle_between(pen_axis(slerp(a, b, 0.5)), [0, 0, -1])
        self.assertAlmostEqual(math.degrees(mid), 15.0, delta=0.05)


class PenAxisTests(unittest.TestCase):
    def test_lead_angle_and_direction(self):
        axis = tilted_pen_axis([0, 0, 1], [1, 0, 0], 15.0)
        self.assertAlmostEqual(math.degrees(angle_between(axis, [0, 0, -1])), 15.0, places=9)
        self.assertGreater(axis[0], 0.0)  # top trails, so the tip leads along +travel

    def test_side_angle_leans_top_left(self):
        axis = tilted_pen_axis([0, 0, 1], [1, 0, 0], 0.0, 10.0)
        self.assertLess(axis[1], 0.0)  # top leans to +y (left of +x travel), axis points the other way
        self.assertAlmostEqual(math.degrees(angle_between(axis, [0, 0, -1])), 10.0, places=9)

    def test_no_travel_keeps_pen_on_normal(self):
        np.testing.assert_allclose(tilted_pen_axis([0, 0, 1], [0, 0, 0], 20.0), [0, 0, -1])

    def test_tangents_fade_at_hairpin(self):
        path = np.r_[np.c_[np.linspace(0, 0.02, 41), np.zeros(41)], np.c_[np.linspace(0.0195, 0, 40), np.zeros(40)]]
        tangents = smoothed_tangents(np.c_[path, np.zeros(len(path))], 0.004)
        self.assertGreater(np.linalg.norm(tangents[5]), 0.9)
        self.assertLess(np.linalg.norm(tangents[40]), 0.3)  # the turn point averages both directions


class SurfaceTests(unittest.TestCase):
    def test_tilted_plane_round_trip_and_low_edge(self):
        plane = TiltedPlane((0.5, 0.1), 0.05, tilt_deg=(6.0, 8.0), size_xy=(0.16, 0.12))
        for uv in ([0.0, 0.0], [0.07, -0.05], [-0.08, 0.06]):
            np.testing.assert_allclose(plane.uv_of(plane.point(uv)), uv, atol=1e-12)
        corners = [plane.point([sx * 0.08, sy * 0.06]) for sx in (-1, 1) for sy in (-1, 1)]
        self.assertAlmostEqual(min(c[2] for c in corners), 0.05, places=12)
        self.assertAlmostEqual(math.degrees(angle_between(plane.normal([0, 0]), [0, 0, 1])), 10.0, delta=0.05)

    def test_height_above_and_ray_hit(self):
        for surface in (TiltedPlane((0.5, 0.1), 0.05, tilt_deg=(6.0, 8.0)),
                        Bumps((0.5, 0.1), 0.05, bumps=((0.0, 0.0, 0.006, 0.015),))):
            uv = np.array([0.012, -0.004])
            point = surface.point(uv)
            self.assertAlmostEqual(surface.height_above(point + 0.002 * surface.normal(uv)), 0.002, delta=2e-6)
            hit = surface.intersect(point + np.array([0.01, -0.02, 0.6]), np.array([-0.01, 0.02, -0.6]))
            np.testing.assert_allclose(hit, point, atol=1e-7)

    def test_bump_peak_normal_is_vertical(self):
        bumps = make_surface({"kind": "bumps", "bumps": [{"center_uv_m": [0.01, 0.0], "height_m": 0.005,
                                                          "sigma_m": 0.01}]}, center_xy=(0, 0), sheet_z=0.0,
                             size_xy=(0.16, 0.12))
        np.testing.assert_allclose(bumps.normal([0.01, 0.0]), [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(bumps.point([0.01, 0.0])[2], 0.005)


class SpringTests(unittest.TestCase):
    def test_force_ink_and_travel(self):
        pen = SpringPen(stiffness_n_per_m=500, preload_n=0.3, travel_m=0.005, ink_threshold_n=0.2)
        self.assertEqual(pen.force(-0.001), 0.0)
        self.assertFalse(pen.inked(pen.force(-0.001)))
        self.assertAlmostEqual(pen.force(0.0024), 1.5)
        self.assertAlmostEqual(pen.compression_for(1.5), 0.0024)
        self.assertAlmostEqual(pen.force(0.009), pen.max_force_n)
        self.assertTrue(pen.bottomed_out(0.009))
        self.assertGreater(pen.ink_width(2.0), pen.ink_width(0.5))

    def test_profiles(self):
        self.assertEqual(force_profile(None, 0.5, 0.01), None)
        self.assertAlmostEqual(force_profile({"profile": "ramp", "start_n": 1, "end_n": 2}, 0.25, 0.0), 1.25)
        sine = {"profile": "sine", "mean_n": 1.5, "amplitude_n": 0.5, "period_m": 0.04}
        self.assertAlmostEqual(force_profile(sine, 0.0, 0.01), 2.0)


class ChallengeTests(unittest.TestCase):
    def test_flat_challenge_reproduces_plain_targets(self):
        for name in CONFIGS:
            config, phases = _plan(name)
            challenge = Challenge.load("flat", config["drawing"])
            self.assertTrue(challenge.is_identity)
            plain = flatten_desired_targets(phases)
            moved = flatten_desired_targets(challenge.apply(phases))
            np.testing.assert_allclose([t.position for t in moved], [t.position for t in plain], atol=1e-12)
            for target in moved:
                np.testing.assert_allclose(target.orientation_wxyz, config["drawing"]["orientation_wxyz"], atol=1e-12)

    def test_twin_protrusion_flat_is_unchanged(self):
        config, phases = _plan("config/xarm7_drawing_real.yaml")
        challenge = Challenge.load("flat", config["drawing"], protrusion_m=0.035)
        moved = flatten_desired_targets(challenge.apply(phases))
        np.testing.assert_allclose([t.position for t in moved], [t.position for t in flatten_desired_targets(phases)],
                                   atol=1e-12)

    def test_tilted_tips_touch_the_plane(self):
        config, phases = _plan("config/xarm7_drawing_real.yaml")
        for protrusion in (0.0, 0.035):
            challenge = Challenge.load("tilted_8_6_lead_10", config["drawing"], protrusion_m=protrusion)
            for phase in challenge.apply(phases):
                for target in phase.targets:
                    tip = target.position + protrusion * pen_axis(target.orientation_wxyz)
                    height = challenge.surface.height_above(tip) - challenge.contact_offset_m
                    if target.pen_down:
                        self.assertAlmostEqual(height, 0.0, delta=1e-9)
                        lean = math.degrees(angle_between(pen_axis(target.orientation_wxyz),
                                                          -challenge.surface.normal([0, 0])))
                        self.assertLessEqual(lean, 10.0 + 1e-6)
                    else:
                        self.assertGreaterEqual(height, -1e-9)

    def test_spring_presses_draw_targets(self):
        config, phases = _plan("config/xarm7_drawing.yaml")
        challenge = Challenge.load("spring_constant", config["drawing"])
        self.assertFalse(challenge.sheet_collision)
        for phase in challenge.apply(phases):
            for target in phase.targets:
                if target.pen_down:
                    self.assertAlmostEqual(target.force_n, 1.5)
                    self.assertAlmostEqual(challenge.surface.height_above(target.position) - challenge.contact_offset_m,
                                           -0.0024, delta=1e-9)

    def test_measure_and_summary_on_perfect_tracking(self):
        config, phases = _plan("config/xarm7_drawing.yaml")
        challenge = Challenge.load("tilted_lead_spring_ramp", config["drawing"])
        rows = []
        for phase in challenge.apply(phases):
            for target in phase.targets:
                if not target.pen_down:
                    continue
                sample = challenge.measure(target.position, matrix_from_quat(target.orientation_wxyz), target)
                self.assertAlmostEqual(sample["tilt_error_deg"], 0.0, delta=1e-6)
                self.assertAlmostEqual(sample["pen_force_n"], target.force_n, delta=1e-6)
                rows.append({"pen_down": 1, **sample})
        summary = challenge.summary(rows)
        self.assertAlmostEqual(summary["ink"]["coverage_fraction"], 1.0)
        self.assertAlmostEqual(summary["ink"]["gap_length_m"], 0.0)
        self.assertLess(summary["surface_path_error"]["desired_to_executed"]["max_error_m"], 1e-6)
        self.assertLess(summary["pen_force_n"]["target_error_rmse_n"], 1e-6)
        self.assertEqual(summary["pen_force_n"]["in_band_fraction"], 1.0)

    def test_lifted_pen_leaves_a_gap(self):
        config, phases = _plan("config/xarm7_drawing.yaml")
        challenge = Challenge.load("spring_constant", config["drawing"])
        rows, lifted = [], []
        for phase in challenge.apply(phases):
            draw = [t for t in phase.targets if t.pen_down]
            if draw:
                lifted.append(float(np.linalg.norm(np.diff([t.position for t in draw[40:60]], axis=0), axis=1).sum()))
            for index, target in enumerate(draw):
                position = target.position.copy()
                if 40 <= index < 60:  # these samples hover 1 mm above the touching height
                    position[2] += 0.0024 + 0.001
                rows.append({"pen_down": 1, **challenge.measure(position, matrix_from_quat(target.orientation_wxyz),
                                                                 target)})
        summary = challenge.summary(rows)
        self.assertLess(summary["ink"]["coverage_fraction"], 1.0)
        # The ends of a lifted run sit within the 1 mm coverage tolerance of the ink on either side.
        tolerance = challenge.coverage_tolerance_m
        self.assertGreater(summary["ink"]["longest_gap_m"], max(lifted) - 2 * tolerance - 0.0005)
        self.assertLess(summary["ink"]["longest_gap_m"], max(lifted) + 0.0005)

    def test_policies_are_refused(self):
        config, _ = _plan("config/xarm7_drawing.yaml")
        challenge = Challenge.load("pen_lead_15", config["drawing"])
        challenge.check_controller("ik")
        with self.assertRaises(ChallengeError):
            challenge.check_controller("act")

    def test_every_preset_stays_in_the_workspace(self):
        presets = load_presets()["challenges"]
        for name in CONFIGS:
            config, phases = _plan(name, "data/act_test/Cat/69fff8521d33da91.json")
            low, high = np.asarray(config["safety"]["workspace_min_m"]), np.asarray(config["safety"]["workspace_max_m"])
            for preset in presets:
                for protrusion in ((0.0, 0.035) if "real" in name else (0.0,)):
                    targets = flatten_desired_targets(
                        Challenge.load(preset, config["drawing"], protrusion_m=protrusion).apply(phases))
                    positions = np.asarray([t.position for t in targets])
                    self.assertTrue(((positions >= low) & (positions <= high)).all(), (name, preset, protrusion))
                    steps = [angle_between(pen_axis(a.orientation_wxyz), pen_axis(b.orientation_wxyz))
                             for a, b in zip(targets[:-1], targets[1:])]
                    self.assertLess(math.degrees(max(steps)), 5.0, (name, preset))

    def test_home_leg_turns_the_pen_gradually(self):
        config, phases = _plan("config/xarm7_drawing.yaml")
        challenge = Challenge.load("pen_fixed_20", config["drawing"])
        moved = challenge.apply(phases)
        final = moved[0].targets[-1]
        leg = Challenge.home_leg("APPROACH_STROKE_0", final.position + np.array([0.0, 0.0, 0.1]),
                                 np.array([0.0, 1.0, 0.0, 0.0]), final, 0.0015)
        np.testing.assert_allclose(leg.targets[-1].position, final.position)
        np.testing.assert_allclose(leg.targets[-1].orientation_wxyz, final.orientation_wxyz, atol=1e-9)
        self.assertIsInstance(leg.targets[0], MotionTarget)
        first = math.degrees(angle_between(pen_axis(leg.targets[0].orientation_wxyz), [0, 0, -1]))
        self.assertLess(first, 1.0)


if __name__ == "__main__":
    unittest.main()
