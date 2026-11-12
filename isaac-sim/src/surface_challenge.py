"""Surface challenges: draw on a tilted or bumpy sheet, with a tilted pen and a pressure target.

A challenge is a named preset in config/surface_challenges.yaml. It changes only the motion
targets, never the planner: build_motion_sequence() still plans the flat drawing (heights above
the configured pen-down height), and Challenge.apply() moves every target onto the surface and
gives it a pen orientation and, while drawing, a force target. Pure numpy, no Isaac imports, so
everything here is unit tested offline; the runners only call apply(), home_leg(), measure() and
summary().

Heights. With pen protrusion d (IK frame to pen tip along the pen axis; 0 when the IK frame is
the pen tip, as in the headless runner) the sheet is at
    sheet_z = pen_down_z - d          when d > 0 (the MCP lab-arm twin, see src/mcp_scene.py)
    sheet_z = paper_top_z             otherwise
and the pen tip's touching height above the sheet is contact_offset = (pen_down_z - d) - sheet_z
(1 mm on the default sim, whose stylus draws just above its paper). A flat challenge with no
tilt and no force target reproduces the original targets exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from drawing_state_machine import MotionTarget, Phase
from metrics import nearest_distances, summarize_errors
from paper_surface import (
    WORLD_X,
    Surface,
    angle_between,
    make_surface,
    orientation_from_axis,
    pen_axis,
    slerp,
    smoothed_tangents,
    tilted_pen_axis,
)
from pen_spring import SpringPen, force_profile

DEFAULT_PRESETS = Path(__file__).resolve().parents[1] / "config" / "surface_challenges.yaml"
SUPPORTED_CONTROLLERS = ("ik", "ik_stream")

ROW_FIELDS = [
    "pen_tip_x_m", "pen_tip_y_m", "pen_tip_z_m", "paper_u_m", "paper_v_m",
    "height_above_contact_m", "compression_m", "pen_force_n", "target_force_n",
    "inked", "bottomed_out", "tilt_error_deg",
]


class ChallengeError(ValueError):
    pass


def load_presets(path: Optional[Path] = None) -> Dict[str, Any]:
    import yaml

    source = Path(path) if path else DEFAULT_PRESETS
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("challenges"), Mapping):
        raise ChallengeError(f"{source} has no 'challenges' mapping")
    return data


def _nearest(reference: np.ndarray, samples: np.ndarray) -> np.ndarray:
    return nearest_distances(reference, samples)


@dataclass
class Challenge:
    name: str
    surface: Surface
    size_xy: tuple
    pen_down_z: float
    contact_offset_m: float
    protrusion_m: float = 0.0
    tilt_mode: str = "none"            # none | travel | fixed
    lead_deg: float = 0.0
    side_deg: float = 0.0
    top_azimuth_deg: float = 180.0     # fixed mode: world direction the pen top leans toward
    follow_normal: bool = True
    tangent_smoothing_m: float = 0.006
    spring: Optional[SpringPen] = None
    force_spec: Optional[Mapping] = None
    ink_tolerance_m: float = 0.0005
    coverage_tolerance_m: float = 0.001
    force_band_n: float = 0.5
    sheet_collision: bool = True
    reference_x: np.ndarray = field(default_factory=lambda: WORLD_X.copy())
    desired_uv_by_stroke: Dict[int, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------ construction

    @classmethod
    def load(cls, name: str, drawing: Mapping, *, protrusion_m: float = 0.0,
             presets_path: Optional[Path] = None) -> "Challenge":
        data = load_presets(presets_path)
        presets = data["challenges"]
        if name not in presets:
            raise ChallengeError(f"Unknown challenge {name!r}; presets: {', '.join(sorted(presets))}")
        spec = {**(data.get("defaults") or {}), **(presets[name] or {})}
        return cls.from_spec(name, spec, drawing, protrusion_m=protrusion_m)

    @classmethod
    def from_spec(cls, name: str, spec: Mapping, drawing: Mapping, *, protrusion_m: float = 0.0) -> "Challenge":
        base_q = np.asarray(drawing.get("orientation_wxyz", [0.0, 1.0, 0.0, 0.0]), dtype=np.float64)
        if angle_between(pen_axis(base_q), [0.0, 0.0, -1.0]) > 0.05:
            raise ChallengeError("Surface challenges assume the configured pen points straight down")
        reference_x = _rotation_x_column(base_q)
        pen_down_z = float(drawing["pen_down_z_m"])
        d = float(protrusion_m)
        sheet_z = pen_down_z - d if d > 0 else float(drawing["paper_top_z_m"])
        size_xy = tuple(float(v) for v in drawing["surface_size_xy_m"])
        surface = make_surface(spec.get("surface"), center_xy=drawing["surface_center_xy_m"], sheet_z=sheet_z,
                               size_xy=size_xy)
        tilt = dict(spec.get("pen_tilt") or {})
        mode = str(tilt.get("mode", "travel" if tilt else "none")).lower()
        if mode not in ("none", "travel", "fixed"):
            raise ChallengeError(f"pen_tilt.mode must be none, travel or fixed, got {mode!r}")
        spring = SpringPen.from_mapping(spec.get("pen_spring"))
        force_spec = spec.get("pen_force") or None
        if force_spec and spring is None:
            raise ChallengeError("pen_force needs a pen_spring to turn force into pen compression")
        return cls(
            name=name,
            surface=surface,
            size_xy=size_xy,
            pen_down_z=pen_down_z,
            contact_offset_m=(pen_down_z - d) - sheet_z,
            protrusion_m=d,
            tilt_mode=mode,
            lead_deg=float(tilt.get("lead_deg", 0.0)),
            side_deg=float(tilt.get("side_deg", 0.0)),
            top_azimuth_deg=float(tilt.get("top_azimuth_deg", 180.0)),
            follow_normal=bool(tilt.get("follow_normal", True)),
            tangent_smoothing_m=float(spec.get("tangent_smoothing_m", 0.006)),
            spring=spring,
            force_spec=force_spec,
            ink_tolerance_m=float(spec.get("ink_tolerance_m", 0.0005)),
            coverage_tolerance_m=float(spec.get("coverage_tolerance_m", 0.001)),
            force_band_n=float(spec.get("force_band_n", 0.5)),
            # A modelled spring pen presses below the touching height; a rigid colliding sheet
            # would fight that, so the sheet loses its collision when the spring is on.
            sheet_collision=bool(spec.get("sheet_collision", True)) and spring is None,
            reference_x=reference_x,
        )

    @property
    def is_identity(self) -> bool:
        """True when apply() would leave every target where it was."""
        return (self.surface.kind == "flat" and self.tilt_mode == "none" and not self.force_spec)

    def check_controller(self, controller: str) -> None:
        if str(controller).lower() not in SUPPORTED_CONTROLLERS:
            raise ChallengeError(
                f"Surface challenges run with --controller {' or '.join(SUPPORTED_CONTROLLERS)}: the trained "
                f"ACT/SmolVLA policies only know the flat sheet and a vertical pen (retrain them first)"
            )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "surface": self.surface.describe(),
            "pen_tilt": {"mode": self.tilt_mode, "lead_deg": self.lead_deg, "side_deg": self.side_deg,
                         "top_azimuth_deg": self.top_azimuth_deg, "follow_normal": self.follow_normal},
            "pen_spring": None if self.spring is None else self.spring.__dict__.copy(),
            "pen_force": dict(self.force_spec) if self.force_spec else None,
            "contact_offset_m": self.contact_offset_m,
            "protrusion_m": self.protrusion_m,
            "sheet_collision": self.sheet_collision,
        }

    # ------------------------------------------------------------ targets

    def _axis(self, uv, travel, fade: float = 1.0) -> np.ndarray:
        n = self.surface.normal(uv) if self.follow_normal else np.array([0.0, 0.0, 1.0])
        if self.tilt_mode == "none":
            return -n
        if self.tilt_mode == "fixed":
            az = np.radians(self.top_azimuth_deg)
            travel = -np.array([np.cos(az), np.sin(az), 0.0])  # top leans toward az, so the tip leads the other way
            fade = 1.0
        return tilted_pen_axis(n, travel, self.lead_deg * fade, self.side_deg * fade)

    def _orientation(self, axis) -> np.ndarray:
        return orientation_from_axis(axis, self.reference_x)

    def _ik_point(self, tip: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        return tip - self.protrusion_m * pen_axis(orientation)

    def apply(self, phases: Sequence[Phase]) -> List[Phase]:
        """Move every target onto the surface; give it a pen orientation and (drawing) a force target."""
        draw_by_stroke: Dict[int, Phase] = {}
        for phase in phases:
            if phase.state.startswith("DRAW_STROKE_"):
                draw_by_stroke[int(phase.targets[0].stroke_id)] = phase

        draw_q: Dict[int, List[np.ndarray]] = {}
        draw_press: Dict[int, List[float]] = {}
        new_draw: Dict[int, Phase] = {}
        self.desired_uv_by_stroke = {}
        for stroke_id, phase in draw_by_stroke.items():
            xy = np.asarray([t.position[:2] for t in phase.targets], dtype=np.float64)
            uv = xy - np.asarray(self.surface.center_xy)
            contact = self.surface.points(uv)
            tangents = smoothed_tangents(contact, self.tangent_smoothing_m)
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(contact, axis=0), axis=1))]
            total = float(arc[-1]) if arc[-1] > 0 else 1.0
            quats, presses, targets = [], [], []
            previous_q = None
            for index, target in enumerate(phase.targets):
                fade = float(min(1.0, np.linalg.norm(tangents[index])))
                q = self._orientation(self._axis(uv[index], tangents[index], fade))
                if previous_q is not None and float(np.dot(q, previous_q)) < 0.0:
                    q = -q
                previous_q = q
                force = force_profile(self.force_spec, arc[index] / total, arc[index])
                press = self.spring.compression_for(force) if (self.spring is not None and force is not None) else 0.0
                normal = self.surface.normal(uv[index])
                tip = contact[index] + normal * (self.contact_offset_m - press)
                targets.append(MotionTarget(target.state, self._ik_point(tip, q), target.stroke_id,
                                            target.waypoint_index, target.pen_down, orientation_wxyz=q,
                                            force_n=force))
                quats.append(q)
                presses.append(press)
            draw_q[stroke_id] = quats
            draw_press[stroke_id] = presses
            new_draw[stroke_id] = Phase(phase.state, targets)
            self.desired_uv_by_stroke[stroke_id] = uv

        out: List[Phase] = []
        previous_stroke: Optional[int] = None
        for phase in phases:
            stroke_id = int(phase.targets[0].stroke_id)
            if phase.state.startswith("DRAW_STROKE_"):
                out.append(new_draw[stroke_id])
                previous_stroke = stroke_id
                continue
            first_q = draw_q[stroke_id][0]
            press_start = press_end = 0.0
            if phase.state.startswith("LIFT_PEN_STROKE_"):
                start_q = end_q = draw_q[stroke_id][-1]
                press_start = draw_press[stroke_id][-1]   # release the spring on the way up
            elif phase.state.startswith("MOVE_TO_STROKE_") and previous_stroke in draw_q:
                start_q, end_q = draw_q[previous_stroke][-1], first_q
            else:  # APPROACH and LOWER use the stroke's first drawing orientation
                start_q = end_q = first_q
                if phase.state.startswith("LOWER_PEN_STROKE_"):
                    press_end = draw_press[stroke_id][0]  # build up to the first force on the way down
            count = len(phase.targets)
            targets = []
            for index, target in enumerate(phase.targets):
                fraction = (index + 1) / count
                q = slerp(start_q, end_q, fraction)
                press = press_start + (press_end - press_start) * fraction
                height = float(target.position[2]) - self.pen_down_z
                uv = np.asarray(target.position[:2], dtype=np.float64) - np.asarray(self.surface.center_xy)
                tip = self.surface.point(uv) + self.surface.normal(uv) * (self.contact_offset_m + height - press)
                targets.append(MotionTarget(target.state, self._ik_point(tip, q), target.stroke_id,
                                            target.waypoint_index, target.pen_down, orientation_wxyz=q))
            out.append(Phase(phase.state, targets))
        return out

    @staticmethod
    def home_leg(state: str, home_point: Sequence[float], home_orientation: Sequence[float],
                 final: MotionTarget, max_step: float) -> Phase:
        """Approach from the settled home pose to the first approach target, turning the pen on the way."""
        from coordinate_mapper import interpolate_segment

        points = interpolate_segment(home_point, final.position, max_step)[1:]
        end_q = final.orientation_wxyz if final.orientation_wxyz is not None else np.asarray(home_orientation)
        count = len(points)
        return Phase(state, [
            MotionTarget(state, point, final.stroke_id, index, False,
                         orientation_wxyz=slerp(home_orientation, end_q, (index + 1) / count))
            for index, point in enumerate(points)
        ])

    def desired_tip_strokes(self) -> Dict[int, np.ndarray]:
        """Pen-tip path per stroke that the drawing asks for (on the sheet, at the touching height).
        Valid after apply()."""
        return {
            stroke_id: self.surface.points(uv) + self.surface.normals(uv) * self.contact_offset_m
            for stroke_id, uv in self.desired_uv_by_stroke.items()
        }

    # ------------------------------------------------------------ measuring a run

    def measure(self, ik_position: Sequence[float], ik_rotation: np.ndarray, target: MotionTarget) -> Dict[str, float]:
        """Per-sample pen state: tip, paper position, height, spring force, ink and tilt error."""
        rotation = np.asarray(ik_rotation, dtype=np.float64).reshape(3, 3)
        axis = rotation[:, 2]
        tip = np.asarray(ik_position, dtype=np.float64) + self.protrusion_m * axis
        uv = self.surface.uv_of(tip)
        height = self.surface.height_above(tip) - self.contact_offset_m
        compression = -height
        if self.spring is not None:
            force = self.spring.force(compression)
            inked = self.spring.inked(force)
            bottomed = self.spring.bottomed_out(compression)
        else:
            force = float("nan")
            inked = height <= self.ink_tolerance_m
            bottomed = False
        desired_q = target.orientation_wxyz
        desired_axis = pen_axis(desired_q) if desired_q is not None else -self.surface.normal(uv)
        return {
            "pen_tip_x_m": float(tip[0]), "pen_tip_y_m": float(tip[1]), "pen_tip_z_m": float(tip[2]),
            "paper_u_m": float(uv[0]), "paper_v_m": float(uv[1]),
            "height_above_contact_m": float(height),
            "compression_m": float(compression),
            "pen_force_n": float(force),
            "target_force_n": float(target.force_n) if target.force_n is not None else float("nan"),
            "inked": int(bool(inked)),
            "bottomed_out": int(bool(bottomed)),
            "tilt_error_deg": float(np.degrees(angle_between(axis, desired_axis))),
        }

    def summary(self, pen_down_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Challenge metrics over the pen-down samples (rows carrying ROW_FIELDS)."""
        result: Dict[str, Any] = {"challenge": self.describe()}
        rows = [row for row in pen_down_rows if "paper_u_m" in row]
        if not rows or not self.desired_uv_by_stroke:
            result["status"] = "no pen-down samples"
            return result
        desired = np.vstack(list(self.desired_uv_by_stroke.values()))
        executed = np.asarray([[row["paper_u_m"], row["paper_v_m"]] for row in rows], dtype=np.float64)
        inked_mask = np.asarray([bool(row["inked"]) for row in rows])
        result["pen_down_samples"] = len(rows)
        result["surface_path_error"] = {
            "desired_to_executed": summarize_errors(_nearest(executed, desired)),
            "executed_to_desired": summarize_errors(_nearest(desired, executed)),
        }
        # Ink coverage of the planned path, stroke by stroke, so pen-up hops never count as gaps.
        covered_length = total_length = gap_length = 0.0
        longest_gap = 0.0
        inked_uv = executed[inked_mask]
        for uv in self.desired_uv_by_stroke.values():
            covered = (_nearest(inked_uv, uv) <= self.coverage_tolerance_m) if len(inked_uv) else np.zeros(len(uv), bool)
            seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            weight = 0.5 * (covered[:-1].astype(float) + covered[1:].astype(float))
            covered_length += float((seg * weight).sum())
            total_length += float(seg.sum())
            run = 0.0
            for length, w in zip(seg, weight):
                if w < 1.0:
                    gap_length += float(length * (1.0 - w))
                    run += float(length * (1.0 - w))
                    longest_gap = max(longest_gap, run)
                else:
                    run = 0.0
        result["ink"] = {
            "inked_sample_fraction": float(inked_mask.mean()),
            "coverage_fraction": covered_length / total_length if total_length > 0 else 0.0,
            "gap_length_m": gap_length,
            "longest_gap_m": longest_gap,
            "coverage_tolerance_m": self.coverage_tolerance_m,
        }
        if inked_mask.any():
            result["surface_path_error"]["inked_to_desired"] = summarize_errors(_nearest(desired, executed[inked_mask]))
        heights = np.asarray([row["height_above_contact_m"] for row in rows], dtype=np.float64)
        result["height_above_contact_m"] = {
            "mean": float(heights.mean()), "p5": float(np.percentile(heights, 5)),
            "p95": float(np.percentile(heights, 95)), "max": float(heights.max()), "min": float(heights.min()),
        }
        tilt = np.asarray([row["tilt_error_deg"] for row in rows], dtype=np.float64)
        result["tilt_error_deg"] = {"mean": float(tilt.mean()), "p95": float(np.percentile(tilt, 95)),
                                    "max": float(tilt.max())}
        if self.spring is not None:
            force = np.asarray([row["pen_force_n"] for row in rows], dtype=np.float64)
            target = np.asarray([row["target_force_n"] for row in rows], dtype=np.float64)
            stats = {
                "mean": float(force.mean()), "p5": float(np.percentile(force, 5)),
                "p95": float(np.percentile(force, 95)), "max": float(force.max()),
                "contact_fraction": float((force > 0).mean()),
                "bottomed_out_samples": int(sum(int(row["bottomed_out"]) for row in rows)),
                "max_force_n": self.spring.max_force_n,
            }
            has_target = np.isfinite(target)
            if has_target.any():
                error = force[has_target] - target[has_target]
                stats["target_error_rmse_n"] = float(np.sqrt(np.mean(error ** 2)))
                stats["in_band_fraction"] = float((np.abs(error) <= self.force_band_n).mean())
                stats["band_n"] = self.force_band_n
            result["pen_force_n"] = stats
        return result


def _rotation_x_column(wxyz) -> np.ndarray:
    from paper_surface import matrix_from_quat

    return matrix_from_quat(wxyz)[:, 0]
