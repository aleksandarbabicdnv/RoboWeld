"""Turn a picture of line art on the sheet into ordered pen strokes ("Path A"). Pure numpy.

The camera (or a flatbed scan) sees dark lines on bright paper. extract_strokes() keeps the pixels
that fall on the drawing area, thins the lines to a one-pixel centre line, follows that centre
line from end to end, joins branches that continue straight through junctions, and orders the
strokes to keep pen-up travel short. The result is in paper metres (units "paper_m"), so the
existing controllers draw it exactly where the lines are (src/coordinate_mapper.py).

Two ways to know where a pixel is on the paper:
- ScanMapping: the image is the drawing area itself (a scan, or a template PNG).
- CameraMapping: a pinhole camera looking straight down; every pixel ray is intersected with the
  surface (flat, tilted or bumpy, src/paper_surface.py), so lines on a tilted sheet land where
  they really are on that sheet.

render_strokes() does the reverse (strokes -> template image) for test sheets and unit tests.
No OpenCV, scikit-image or SciPy, so the same code runs in Isaac Sim's Python and in WSL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metrics import nearest_distances

# ---------------------------------------------------------------- pixel <-> paper mappings


@dataclass
class ScanMapping:
    """The image covers the drawing area (size_xy metres) exactly; row 0 is the +v edge."""

    size_xy: Tuple[float, float]

    def pixel_uv(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        height, width = int(shape[0]), int(shape[1])
        sx, sy = float(self.size_xy[0]), float(self.size_xy[1])
        cols = (np.arange(width) + 0.5) / width * sx - sx / 2.0
        rows = sy / 2.0 - (np.arange(height) + 0.5) / height * sy
        uv = np.stack(np.meshgrid(cols, rows), axis=-1)
        return uv, np.ones((height, width), dtype=bool)

    def pixel_size_m(self, shape) -> float:
        return float(self.size_xy[0]) / int(shape[1])


@dataclass
class CameraMapping:
    """Pinhole camera at `eye` looking straight down, image right = world +x, image up = world +y.

    view_width_m is the width the image spans at `distance_m` below the camera (horizontal
    aperture fit, as USD/RTX cameras use); pixels are square.
    """

    eye: Tuple[float, float, float]
    view_width_m: float
    distance_m: float
    surface: object
    size_xy: Tuple[float, float]
    border_m: float = 0.002

    def focal_px(self, width: int) -> float:
        return width * self.distance_m / self.view_width_m

    def rays(self, shape) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        f = self.focal_px(width)
        cols = (np.arange(width) + 0.5 - width / 2.0) / f
        rows = -(np.arange(height) + 0.5 - height / 2.0) / f
        dx, dy = np.meshgrid(cols, rows)
        return np.stack([dx, dy, -np.ones_like(dx)], axis=-1)

    def pixel_uv(self, shape) -> Tuple[np.ndarray, np.ndarray]:
        directions = self.rays(shape).reshape(-1, 3)
        origin = np.asarray(self.eye, dtype=np.float64)
        hits = intersect_many(self.surface, origin, directions)
        uv = uv_of_many(self.surface, hits)
        half = np.asarray(self.size_xy, dtype=np.float64) / 2.0 - self.border_m
        valid = np.isfinite(uv).all(axis=1) & (np.abs(uv[:, 0]) <= half[0]) & (np.abs(uv[:, 1]) <= half[1])
        return uv.reshape(*shape[:2], 2), valid.reshape(shape[:2])

    def pixel_size_m(self, shape) -> float:
        return self.view_width_m / int(shape[1])


def _surface_points(surface, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    kind = getattr(surface, "kind", "flat")
    cx, cy = surface.center_xy
    if kind == "tilted":
        local = np.c_[uv, np.zeros(len(uv))]
        return surface.origin[None, :] + local @ surface.rotation.T
    z = np.full(len(uv), float(surface.sheet_z))
    if kind == "bumps":
        for cu, cv, h, s in surface.bumps:
            z = z + h * np.exp(-((uv[:, 0] - cu) ** 2 + (uv[:, 1] - cv) ** 2) / (2.0 * s * s))
    return np.c_[uv[:, 0] + cx, uv[:, 1] + cy, z]


def uv_of_many(surface, world: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    if getattr(surface, "kind", "flat") == "tilted":
        return ((world - surface.origin[None, :]) @ surface.rotation)[:, :2]
    cx, cy = surface.center_xy
    return np.c_[world[:, 0] - cx, world[:, 1] - cy]


def intersect_many(surface, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """First hit of each ray with the surface (NaN rows for misses). Vectorised version of
    Surface.intersect for all three surface kinds."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    out = np.full_like(d, np.nan)
    kind = getattr(surface, "kind", "flat")
    if kind == "tilted":
        n = surface.rotation[:, 2]
        denom = d @ n
        ok = np.abs(denom) > 1e-12
        s = np.where(ok, ((surface.origin - o) @ n) / np.where(ok, denom, 1.0), -1.0)
        ok &= s > 0
        out[ok] = o + s[ok, None] * d[ok]
        return out
    ok = np.abs(d[:, 2]) > 1e-12
    z = np.full(len(d), float(surface.sheet_z))
    for _ in range(50 if kind == "bumps" else 1):
        s = np.where(ok, (z - o[2]) / np.where(ok, d[:, 2], 1.0), -1.0)
        hit = o + s[:, None] * d
        new_z = _surface_points(surface, uv_of_many(surface, hit))[:, 2]
        if kind != "bumps" or np.nanmax(np.abs(new_z - z)) < 1e-9:
            z = new_z
            break
        z = new_z
    s = np.where(ok, (z - o[2]) / np.where(ok, d[:, 2], 1.0), -1.0)
    ok &= s > 0
    out[ok] = o + s[ok, None] * d[ok]
    return out


# ---------------------------------------------------------------- image processing


def to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return array.astype(np.float64)


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window via an integral image (edges use the part inside)."""
    if radius < 1:
        return image.astype(np.float64)
    a = np.pad(image.astype(np.float64), ((1, 0), (1, 0)))
    integral = a.cumsum(0).cumsum(1)
    ones = np.pad(np.ones(image.shape), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = image.shape
    r0 = np.clip(np.arange(h) - radius, 0, h)
    r1 = np.clip(np.arange(h) + radius + 1, 0, h)
    c0 = np.clip(np.arange(w) - radius, 0, w)
    c1 = np.clip(np.arange(w) + radius + 1, 0, w)

    def window(table):
        return (table[r1][:, c1] - table[r0][:, c1] - table[r1][:, c0] + table[r0][:, c0])

    return window(integral) / np.maximum(window(ones), 1.0)


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    histogram, edges = np.histogram(values, bins=256, range=(float(values.min()), float(values.max()) + 1e-9))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(histogram)
    weight1 = weight0[-1] - weight0
    sum0 = np.cumsum(histogram * centers)
    mean0 = sum0 / np.maximum(weight0, 1)
    mean1 = (sum0[-1] - sum0) / np.maximum(weight1, 1)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[int(np.argmax(between))])


def zhang_suen_thin(mask: np.ndarray, max_iterations: int = 200) -> np.ndarray:
    """One-pixel-wide centre lines of a binary mask (Zhang and Suen, 1984), vectorised."""
    img = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p1 = img[1:-1, 1:-1]
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(r.astype(np.int16) for r in ring[:8])
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                c1, c2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            remove = (p1 == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & c1 & c2
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_paths(skeleton: np.ndarray) -> List[np.ndarray]:
    """Pixel paths (K x 2 arrays of row, col) between ends and junctions of a skeleton; loops closed."""
    skel = np.asarray(skeleton, dtype=bool)
    h, w = skel.shape
    rows, cols = np.nonzero(skel)
    index = -np.ones((h + 2, w + 2), dtype=np.int64)
    index[rows + 1, cols + 1] = np.arange(len(rows))
    padded = np.pad(skel, 1)
    adjacency: List[List[int]] = [[] for _ in range(len(rows))]
    for k, (r, c) in enumerate(zip(rows + 1, cols + 1)):
        for dr, dc in _OFFSETS:
            j = index[r + dr, c + dc]
            if j < 0:
                continue
            if dr != 0 and dc != 0 and (padded[r + dr, c] or padded[r, c + dc]):
                continue  # diagonal shortcut of a 4-connected corner: keep the corner, drop the triangle
            adjacency[k].append(int(j))
    degree = np.array([len(a) for a in adjacency])
    visited = set()

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, nxt):
        path = [start, nxt]
        visited.add(edge(start, nxt))
        current = nxt
        while degree[current] == 2:
            options = [n for n in adjacency[current] if edge(current, n) not in visited]
            if not options:
                break
            following = options[0]
            visited.add(edge(current, following))
            path.append(following)
            current = following
        return path

    paths = []
    for k in np.nonzero(degree != 2)[0]:
        for n in adjacency[k]:
            if edge(k, n) not in visited:
                paths.append(walk(int(k), n))
    for k in np.nonzero(degree == 2)[0]:
        for n in adjacency[k]:
            if edge(k, n) not in visited:
                paths.append(walk(int(k), n))
    return [np.c_[rows[p], cols[p]] for p in paths if len(p) >= 2]


# ---------------------------------------------------------------- polylines


def polyline_length(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def smooth_polyline(points: np.ndarray, window: int = 5, closed: bool = False) -> np.ndarray:
    """Moving average that removes pixel staircases; open ends stay fixed."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3 or window < 3:
        return p.copy()
    half = window // 2
    if closed:
        core = p[:-1]
        padded = np.vstack([core[-half:], core, core[:half]])
        kernel = np.ones(window) / window
        out = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(p.shape[1])])
        return np.vstack([out, out[:1]])
    out = p.copy()
    for i in range(1, len(p) - 1):
        k = min(half, i, len(p) - 1 - i)
        out[i] = p[i - k:i + k + 1].mean(axis=0)
    return out


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iterative."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3 or tolerance <= 0:
        return p.copy()
    keep = np.zeros(len(p), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = p[b] - p[a]
        length = np.linalg.norm(seg)
        rel = p[a + 1:b] - p[a]
        if length < 1e-12:
            dist = np.linalg.norm(rel, axis=1)
        else:
            dist = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[a + 1 + k] = True
            stack.extend([(a, a + 1 + k), (a + 1 + k, b)])
    return p[keep]


def _end_direction(points: np.ndarray, at_start: bool, reach: float) -> np.ndarray:
    """Unit direction pointing OUT of the stroke at one end, measured over `reach` metres."""
    p = points if not at_start else points[::-1]
    end = p[-1]
    run = 0.0
    inner = p[-2]
    for k in range(len(p) - 2, -1, -1):
        run += float(np.linalg.norm(p[k + 1] - p[k]))
        inner = p[k]
        if run >= reach:
            break
    v = end - inner
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def chain_strokes(strokes: List[np.ndarray], join_m: float, max_turn_deg: float = 60.0,
                  reach_m: float = 0.002) -> List[np.ndarray]:
    """Join stroke ends that meet (within join_m) and continue straight on (turn <= max_turn_deg).

    At a T the two arms of the bar join and the stem stays separate; at an X both lines go
    straight through. A stroke whose own ends meet becomes a closed loop.
    """
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    limit = math.cos(math.radians(max_turn_deg))
    while True:
        best = None
        ends = []
        for i, s in enumerate(strokes):
            if np.linalg.norm(s[0] - s[-1]) <= 1e-12:
                continue  # already closed
            ends.append((i, True, s[0], _end_direction(s, True, reach_m)))
            ends.append((i, False, s[-1], _end_direction(s, False, reach_m)))
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                i, start_i, pi, di = ends[a]
                j, start_j, pj, dj = ends[b]
                gap = float(np.linalg.norm(pi - pj))
                if gap > join_m:
                    continue
                straightness = float(np.dot(di, -dj))  # leaving i should continue as entering j
                if i == j:
                    if polyline_length(strokes[i]) < 4 * join_m:
                        continue
                    straightness = 1.0  # closing a loop
                if straightness < limit:
                    continue
                score = straightness - gap / join_m * 0.1
                if best is None or score > best[0]:
                    best = (score, i, start_i, j, start_j)
        if best is None:
            return strokes
        _, i, start_i, j, start_j = best
        if i == j:
            strokes[i] = np.vstack([strokes[i], strokes[i][:1]])
            continue
        a = strokes[i][::-1] if start_i else strokes[i]      # a ends at the joined end
        b = strokes[j] if start_j else strokes[j][::-1]      # b starts at the joined end
        merged = np.vstack([a, b])
        strokes = [s for k, s in enumerate(strokes) if k not in (i, j)] + [merged]


def order_strokes(strokes: List[np.ndarray], start: Sequence[float] = (0.0, 0.0)) -> List[np.ndarray]:
    """Greedy nearest-next ordering; open strokes may be reversed, closed loops start where the pen is."""
    remaining = [np.asarray(s, dtype=np.float64) for s in strokes]
    pen = np.asarray(start, dtype=np.float64)
    ordered = []
    while remaining:
        best = None
        for k, s in enumerate(remaining):
            closed = np.linalg.norm(s[0] - s[-1]) <= 1e-12
            if closed:
                d = np.linalg.norm(s[:-1] - pen, axis=1)
                m = int(np.argmin(d))
                cand = (float(d[m]), k, ("loop", m))
            else:
                d0, d1 = np.linalg.norm(s[0] - pen), np.linalg.norm(s[-1] - pen)
                cand = (float(d0), k, "fwd") if d0 <= d1 else (float(d1), k, "rev")
            if best is None or cand[0] < best[0]:
                best = cand
        _, k, how = best
        s = remaining.pop(k)
        if how == "rev":
            s = s[::-1]
        elif isinstance(how, tuple):
            core = s[:-1]
            s = np.vstack([np.roll(core, -how[1], axis=0), core[how[1]:how[1] + 1]])
        ordered.append(s)
        pen = s[-1]
    return ordered


# ---------------------------------------------------------------- the pipeline


@dataclass
class ExtractionResult:
    strokes: List[np.ndarray]           # paper metres (u, v)
    threshold: float
    ink_pixels: int
    skeleton_pixels: int
    pixel_size_m: float
    raw_paths: int


def extract_strokes(image: np.ndarray, mapping, *, min_length_m: float = 0.003, join_m: Optional[float] = None,
                    max_turn_deg: float = 60.0, smooth_window: int = 5, simplify_m: float = 0.0001,
                    background_radius_m: float = 0.006, min_contrast: float = 40.0,
                    start_uv: Sequence[float] = (-1.0, 1.0)) -> ExtractionResult:
    """Dark lines in `image` -> ordered strokes in paper metres. See the module docstring."""
    gray = to_gray(image)
    uv_map, valid = mapping.pixel_uv(gray.shape)
    if not valid.any():
        raise ValueError("No pixel of the image falls on the drawing area")
    pixel = mapping.pixel_size_m(gray.shape)
    radius = max(1, int(round(background_radius_m / pixel)))
    paper = np.where(valid, gray, np.nan)
    fill = float(np.nanmedian(paper))
    # Divide out slow light changes (shading on a tilted or bumpy sheet) before thresholding.
    background = box_blur(np.where(valid, gray, fill), radius)
    background = np.maximum(background, box_blur(background, radius))
    normalized = np.clip(gray / np.maximum(background, 1.0) * 255.0, 0, 255)
    values = normalized[valid]
    if float(np.percentile(values, 99.5) - np.percentile(values, 0.5)) < min_contrast:
        raise ValueError("The drawing area shows no lines (contrast too low)")
    threshold = otsu_threshold(values)
    ink = valid & (normalized < threshold)
    skeleton = zhang_suen_thin(ink)
    paths = skeleton_paths(skeleton)
    strokes = []
    for path in paths:
        uv = uv_map[path[:, 0], path[:, 1]]
        if not np.isfinite(uv).all():
            continue
        strokes.append(uv)
    join = join_m if join_m is not None else 4.0 * pixel
    strokes = chain_strokes([s for s in strokes if polyline_length(s) >= 1.5 * pixel], join, max_turn_deg)
    cleaned = []
    for s in strokes:
        closed = np.linalg.norm(s[0] - s[-1]) <= 1e-12
        s = smooth_polyline(s, smooth_window, closed=closed)
        s = simplify(s, simplify_m)
        if polyline_length(s) >= min_length_m:
            cleaned.append(s)
    ordered = order_strokes(cleaned, start=start_uv)
    return ExtractionResult(ordered, threshold, int(ink.sum()), int(skeleton.sum()), pixel, len(paths))


def strokes_to_trajectory(strokes: Sequence[np.ndarray], drawing_id: str, **extra) -> Dict:
    return {
        "drawing_id": str(drawing_id),
        "units": "paper_m",
        "strokes": [{"stroke_id": k, "points": np.asarray(s, dtype=np.float64).round(6).tolist()}
                    for k, s in enumerate(strokes)],
        **extra,
    }


# ---------------------------------------------------------------- making and scoring templates


def render_strokes(strokes_uv: Sequence[np.ndarray], size_xy: Sequence[float], px_per_m: float,
                   width_m: float = 0.0008) -> np.ndarray:
    """Black lines on white, (H, W) uint8, covering the drawing area; row 0 is the +v edge."""
    sx, sy = float(size_xy[0]), float(size_xy[1])
    width_px, height_px = int(round(sx * px_per_m)), int(round(sy * px_per_m))
    image = np.full((height_px, width_px), 255, dtype=np.uint8)
    half = 0.5 * width_m * px_per_m
    for stroke in strokes_uv:
        p = np.asarray(stroke, dtype=np.float64)
        px = np.c_[(p[:, 0] + sx / 2.0) * px_per_m - 0.5, (sy / 2.0 - p[:, 1]) * px_per_m - 0.5]
        for a, b in zip(px[:-1], px[1:]):
            c0 = int(max(0, math.floor(min(a[0], b[0]) - half - 1)))
            c1 = int(min(width_px, math.ceil(max(a[0], b[0]) + half + 2)))
            r0 = int(max(0, math.floor(min(a[1], b[1]) - half - 1)))
            r1 = int(min(height_px, math.ceil(max(a[1], b[1]) + half + 2)))
            if c0 >= c1 or r0 >= r1:
                continue
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            seg = b - a
            length2 = float(seg @ seg)
            t = np.zeros_like(cc, dtype=np.float64) if length2 < 1e-12 else \
                np.clip(((cc - a[0]) * seg[0] + (rr - a[1]) * seg[1]) / length2, 0.0, 1.0)
            dist = np.hypot(cc - (a[0] + t * seg[0]), rr - (a[1] + t * seg[1]))
            image[r0:r1, c0:c1][dist <= half] = 0
    return image


def resample(points: np.ndarray, step: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] <= 0:
        return p[:1]
    samples = np.linspace(0.0, arc[-1], max(2, int(math.ceil(arc[-1] / step)) + 1))
    return np.c_[np.interp(samples, arc, p[:, 0]), np.interp(samples, arc, p[:, 1])]


def compare_strokes(extracted: Sequence[np.ndarray], truth: Sequence[np.ndarray], step_m: float = 0.00025) -> Dict:
    """How well extracted strokes trace the true lines (both directions, in metres)."""
    a = np.vstack([resample(s, step_m) for s in extracted]) if extracted else np.zeros((0, 2))
    b = np.vstack([resample(s, step_m) for s in truth])
    if len(a) == 0:
        return {"status": "no strokes extracted"}
    truth_to_extracted = nearest_distances(a, b)
    extracted_to_truth = nearest_distances(b, a)
    return {
        "truth_to_extracted_mean_m": float(truth_to_extracted.mean()),
        "truth_to_extracted_p95_m": float(np.percentile(truth_to_extracted, 95)),
        "extracted_to_truth_mean_m": float(extracted_to_truth.mean()),
        "extracted_to_truth_p95_m": float(np.percentile(extracted_to_truth, 95)),
        "coverage_at_0_5mm": float((truth_to_extracted <= 0.0005).mean()),
        "coverage_at_1mm": float((truth_to_extracted <= 0.001).mean()),
        "extracted_length_m": float(sum(polyline_length(s) for s in extracted)),
        "truth_length_m": float(sum(polyline_length(s) for s in truth)),
        "extracted_strokes": len(extracted),
        "truth_strokes": len(truth),
    }
