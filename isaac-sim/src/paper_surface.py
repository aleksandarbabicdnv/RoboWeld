"""Shape of the drawing sheet and the pen's orientation over it. Pure numpy, no Isaac imports.

A surface is described in paper coordinates (u, v): metres from the centre of the drawing area,
+u along world +x and +v along world +y when the sheet is flat. Every surface answers the same
questions: where is paper point (u, v) in the world, what is the outward normal there, how high
is a world point above the sheet (along the normal), and where does a camera ray hit the sheet.

- flat:   the original sheet, z = sheet_z everywhere.
- tilted: a rigid plane rotated about world x and then world y. By default it is lifted so its
          lowest corner stays on the original sheet height, as if a wedge were slid under it.
          (u, v) lie in the plane, so drawings keep their true size on the tilted sheet.
- bumps:  a height field z = sheet_z + sum of Gaussian bumps; (u, v) are the world x, y offsets
          (the drawing is projected straight down onto the bumps).

Pen orientation: the pen axis is the IK frame's +Z, pointing from the pen body into the paper.
The flat default orientation_wxyz [0, 1, 0, 0] (180 deg about world x) is exactly "axis = -normal,
frame x = world x". Tilt is applied as a lead angle (the pen top trails the direction of travel,
like normal handwriting or a drag weld) and a side angle (lean to the left of travel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

WORLD_X = np.array([1.0, 0.0, 0.0])
WORLD_Y = np.array([0.0, 1.0, 0.0])
WORLD_Z = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- rotations and quaternions

def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix, with w >= 0."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def matrix_from_quat(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(wxyz, dtype=np.float64) / np.linalg.norm(wxyz))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: Sequence[float], q1: Sequence[float], fraction: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + fraction * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(1.0, dot))
    out = (math.sin((1.0 - fraction) * theta) * a + math.sin(fraction * theta) * b) / math.sin(theta)
    return out / np.linalg.norm(out)


def pen_axis(wxyz: Sequence[float]) -> np.ndarray:
    """World direction of the IK frame's +Z (the pen axis, pointing into the paper)."""
    return matrix_from_quat(wxyz)[:, 2]


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def orientation_from_axis(axis: Sequence[float], reference_x: Sequence[float] = WORLD_X) -> np.ndarray:
    """wxyz of a frame whose +Z is `axis` and whose +X stays as close to `reference_x` as possible.

    The pen is round, so only the axis matters physically; pinning +X near world x keeps the wrist
    from spinning between neighbouring waypoints, and gives back [0, 1, 0, 0] for a vertical pen.
    """
    z = np.asarray(axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    ref = np.asarray(reference_x, dtype=np.float64)
    x = ref - np.dot(ref, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = WORLD_Y - np.dot(WORLD_Y, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return quat_from_matrix(np.column_stack([x, y, z]))


def tilted_pen_axis(normal: Sequence[float], travel: Optional[Sequence[float]], lead_deg: float,
                    side_deg: float = 0.0) -> np.ndarray:
    """Pen axis for a given outward surface normal and in-surface travel direction.

    lead_deg > 0 leans the pen top back against the direction of travel (the tip is dragged);
    side_deg > 0 leans the pen top to the left of travel. With no travel direction (or a zero
    one) the pen stays along -normal.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if travel is None or (lead_deg == 0.0 and side_deg == 0.0):
        return -n
    t = np.asarray(travel, dtype=np.float64)
    t = t - np.dot(t, n) * n
    norm = np.linalg.norm(t)
    if norm < 1e-9:
        return -n
    t /= norm
    left = np.cross(n, t)
    lead, side = math.radians(lead_deg), math.radians(side_deg)
    # The pen top sits at tip - L * axis, so leaning the top back (-t) and left (+left) puts +t and
    # -left into the axis.
    axis = -n * math.cos(lead) * math.cos(side) + t * math.sin(lead) * math.cos(side) - left * math.sin(side)
    return axis / np.linalg.norm(axis)


# ---------------------------------------------------------------- surfaces

@dataclass
class Surface:
    """Base class: the flat sheet at `sheet_z`, centred on `center_xy`."""

    center_xy: Tuple[float, float]
    sheet_z: float
    kind: str = "flat"

    def point(self, uv: Sequence[float]) -> np.ndarray:
        u, v = float(uv[0]), float(uv[1])
        return np.array([self.center_xy[0] + u, self.center_xy[1] + v, self.sheet_z])

    def normal(self, uv: Sequence[float]) -> np.ndarray:
        return WORLD_Z.copy()

    def uv_of(self, world: Sequence[float]) -> np.ndarray:
        """Paper coordinates of the sheet point nearest (below/above) a world point."""
        p = np.asarray(world, dtype=np.float64)
        return np.array([p[0] - self.center_xy[0], p[1] - self.center_xy[1]])

    def height_above(self, world: Sequence[float]) -> float:
        """Signed distance of a world point above the sheet, along the local normal."""
        p = np.asarray(world, dtype=np.float64)
        uv = self.uv_of(p)
        return float(np.dot(p - self.point(uv), self.normal(uv)))

    def intersect(self, origin: Sequence[float], direction: Sequence[float]) -> Optional[np.ndarray]:
        """World point where a ray first meets the sheet (None if it misses)."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if abs(d[2]) < 1e-12:
            return None
        s = (self.sheet_z - o[2]) / d[2]
        return None if s <= 0 else o + s * d

    def points(self, uv: np.ndarray) -> np.ndarray:
        return np.array([self.point(row) for row in np.asarray(uv, dtype=np.float64).reshape(-1, 2)])

    def normals(self, uv: np.ndarray) -> np.ndarray:
        return np.array([self.normal(row) for row in np.asarray(uv, dtype=np.float64).reshape(-1, 2)])

    def describe(self) -> dict:
        return {"kind": self.kind, "center_xy_m": list(map(float, self.center_xy)), "sheet_z_m": float(self.sheet_z)}


@dataclass
class TiltedPlane(Surface):
    """A rigid plane: world = origin + R @ (u, v, 0)."""

    tilt_deg: Tuple[float, float] = (0.0, 0.0)
    size_xy: Tuple[float, float] = (0.16, 0.12)
    pivot: str = "low_edge"
    kind: str = "tilted"
    rotation: np.ndarray = field(init=False, repr=False)
    origin: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.rotation = rotation_y(math.radians(self.tilt_deg[1])) @ rotation_x(math.radians(self.tilt_deg[0]))
        self.origin = np.array([self.center_xy[0], self.center_xy[1], self.sheet_z], dtype=np.float64)
        if self.pivot == "low_edge":
            hx, hy = self.size_xy[0] / 2.0, self.size_xy[1] / 2.0
            corners = np.array([[sx * hx, sy * hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
            lowest = float(np.min((corners @ self.rotation.T)[:, 2]))
            self.origin[2] -= lowest  # lowest corner lands on the original sheet height
        elif self.pivot != "center":
            raise ValueError(f"Unknown pivot {self.pivot!r}; use 'low_edge' or 'center'")

    def point(self, uv):
        return self.origin + self.rotation @ np.array([float(uv[0]), float(uv[1]), 0.0])

    def normal(self, uv):
        return self.rotation[:, 2].copy()

    def uv_of(self, world):
        local = self.rotation.T @ (np.asarray(world, dtype=np.float64) - self.origin)
        return local[:2].copy()

    def intersect(self, origin, direction):
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        n = self.rotation[:, 2]
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-12:
            return None
        s = float(np.dot(n, self.origin - o)) / denom
        return None if s <= 0 else o + s * d

    def describe(self):
        return {**super().describe(), "tilt_deg": list(map(float, self.tilt_deg)), "pivot": self.pivot,
                "origin_m": self.origin.tolist(), "normal": self.rotation[:, 2].tolist()}


@dataclass
class Bumps(Surface):
    """Height field: z = sheet_z + sum_i h_i * exp(-|uv - c_i|^2 / (2 s_i^2))."""

    bumps: Tuple[Tuple[float, float, float, float], ...] = ()  # (cu, cv, height, sigma) in metres
    kind: str = "bumps"

    def _terms(self, uv):
        u, v = float(uv[0]), float(uv[1])
        height, du, dv = 0.0, 0.0, 0.0
        for cu, cv, h, s in self.bumps:
            g = h * math.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2.0 * s * s))
            height += g
            du += -g * (u - cu) / (s * s)
            dv += -g * (v - cv) / (s * s)
        return height, du, dv

    def point(self, uv):
        height, _, _ = self._terms(uv)
        return np.array([self.center_xy[0] + float(uv[0]), self.center_xy[1] + float(uv[1]), self.sheet_z + height])

    def normal(self, uv):
        _, du, dv = self._terms(uv)
        n = np.array([-du, -dv, 1.0])
        return n / np.linalg.norm(n)

    def height_above(self, world):
        # First order: vertical gap times the normal's z component (exact on a plane, and the
        # bumps here are gentle). Good to well under a micrometre at pen-contact distances.
        p = np.asarray(world, dtype=np.float64)
        uv = self.uv_of(p)
        n = self.normal(uv)
        return float((p[2] - self.point(uv)[2]) * n[2])

    def intersect(self, origin, direction):
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if abs(d[2]) < 1e-12:
            return None
        z = self.sheet_z
        hit = None
        for _ in range(50):  # fixed point: hit the plane at the current height, update the height
            s = (z - o[2]) / d[2]
            if s <= 0:
                return None
            hit = o + s * d
            new_z = self.point(self.uv_of(hit))[2]
            if abs(new_z - z) < 1e-9:
                break
            z = new_z
        return hit

    def max_slope_deg(self) -> float:
        # Steepest slope of one Gaussian is h / (s * sqrt(e)); a sum is bounded by the sum.
        slope = sum(abs(h) / (s * math.sqrt(math.e)) for _, _, h, s in self.bumps)
        return math.degrees(math.atan(slope))

    def describe(self):
        return {**super().describe(), "bumps_m": [list(map(float, b)) for b in self.bumps],
                "max_slope_deg_bound": self.max_slope_deg()}


def make_surface(spec: Optional[Mapping], *, center_xy: Sequence[float], sheet_z: float,
                 size_xy: Sequence[float]) -> Surface:
    """Build a surface from a challenge's `surface:` mapping (None or kind flat = the original sheet)."""
    spec = dict(spec or {})
    kind = str(spec.get("kind", "flat")).lower()
    center = (float(center_xy[0]), float(center_xy[1]))
    if kind == "flat":
        return Surface(center, float(sheet_z))
    if kind == "tilted":
        tilt = spec.get("tilt_deg", [0.0, 0.0])
        return TiltedPlane(center, float(sheet_z), tilt_deg=(float(tilt[0]), float(tilt[1])),
                           size_xy=(float(size_xy[0]), float(size_xy[1])), pivot=str(spec.get("pivot", "low_edge")))
    if kind == "bumps":
        bumps = []
        for bump in spec.get("bumps", []):
            cu, cv = (float(v) for v in bump["center_uv_m"])
            bumps.append((cu, cv, float(bump["height_m"]), float(bump["sigma_m"])))
            if float(bump["sigma_m"]) <= 0:
                raise ValueError("Bump sigma_m must be positive")
        return Bumps(center, float(sheet_z), bumps=tuple(bumps))
    raise ValueError(f"Unknown surface kind {kind!r}; use flat, tilted or bumps")


# ---------------------------------------------------------------- geometry helpers

def smoothed_tangents(points: np.ndarray, sigma_m: float) -> np.ndarray:
    """Gaussian-weighted (along arc length) average of unit segment directions at each point.

    The result is NOT renormalised: its length drops toward 0 where the path doubles back inside
    the smoothing window, which the pen-tilt code uses to fade the lead angle out at hairpins
    instead of flipping the wrist.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return np.zeros_like(p)
    seg = np.diff(p, axis=0)
    length = np.linalg.norm(seg, axis=1)
    unit = np.divide(seg, length[:, None], out=np.zeros_like(seg), where=length[:, None] > 1e-12)
    mid = np.r_[0.0, np.cumsum(length)]
    seg_mid = 0.5 * (mid[:-1] + mid[1:])
    out = np.zeros_like(p)
    for index, s in enumerate(mid):
        if sigma_m <= 0:
            k = min(index, len(unit) - 1)
            out[index] = unit[k]
            continue
        w = np.exp(-0.5 * ((seg_mid - s) / sigma_m) ** 2) * length
        total = float(w.sum())
        out[index] = (w[:, None] * unit).sum(axis=0) / total if total > 0 else 0.0
    return out


def surface_mesh(surface: Surface, size_xy: Sequence[float], resolution_m: float = 0.004,
                 margin_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid mesh of the sheet: (points Nx3, triangles Mx3 int, uv Nx2 in paper metres)."""
    half = np.asarray(size_xy, dtype=np.float64) / 2.0 + float(margin_m)
    nx = max(2, int(math.ceil(2 * half[0] / resolution_m)) + 1)
    ny = max(2, int(math.ceil(2 * half[1] / resolution_m)) + 1)
    us = np.linspace(-half[0], half[0], nx)
    vs = np.linspace(-half[1], half[1], ny)
    uv = np.array([[u, v] for v in vs for u in us])
    points = surface.points(uv)
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b, c, d = a + 1, a + nx, a + nx + 1
            triangles.append([a, b, d])
            triangles.append([a, d, c])
    return points, np.asarray(triangles, dtype=np.int64), uv


def iter_draw_targets(phases) -> Iterable:
    for phase in phases:
        if phase.state.startswith("DRAW_STROKE_"):
            yield phase
