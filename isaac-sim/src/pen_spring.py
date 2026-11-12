"""Spring-loaded pen holder model: pen force and ink from where the pen tip is. Pure Python.

The arm is position controlled and very stiff (the twin's servo is 750000 N m/rad), so in a
spring-loaded holder the force on the paper is set by how far the holder is pushed past the point
where the tip first touches, not by the arm's dynamics:

    compression x = contact height - height of the uncompressed tip above the sheet (along normal)
    x <= 0          -> pen in the air, force 0, no ink
    0 < x <= travel -> force = preload + stiffness * x
    x > travel      -> the holder bottoms out: rigid contact, flagged as an over-press event

Ink is laid where the force reaches ink_threshold_n; the line widens with force. A pressure
target becomes a commanded compression, (target - preload) / stiffness, i.e. how far below the
touching height the controller aims the uncompressed tip. Nothing here needs a physics contact,
so it also works where the sheet has no collision (the MCP scene, visual-only paper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SpringPen:
    stiffness_n_per_m: float = 500.0
    preload_n: float = 0.3
    travel_m: float = 0.005
    ink_threshold_n: float = 0.2
    ink_width_min_m: float = 0.0004
    ink_width_per_n_m: float = 0.0002
    ink_width_max_m: float = 0.0012

    def __post_init__(self):
        if self.stiffness_n_per_m <= 0 or self.travel_m <= 0:
            raise ValueError("stiffness_n_per_m and travel_m must be positive")
        if self.preload_n < 0 or self.ink_threshold_n < 0:
            raise ValueError("preload_n and ink_threshold_n must be >= 0")

    @classmethod
    def from_mapping(cls, spec: Optional[Mapping]) -> Optional["SpringPen"]:
        if not spec:
            return None
        known = {key: float(value) for key, value in dict(spec).items() if key in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def max_force_n(self) -> float:
        return self.preload_n + self.stiffness_n_per_m * self.travel_m

    def force(self, compression_m: float) -> float:
        x = float(compression_m)
        if x <= 0.0:
            return 0.0
        return self.preload_n + self.stiffness_n_per_m * min(x, self.travel_m)

    def bottomed_out(self, compression_m: float) -> bool:
        return float(compression_m) > self.travel_m

    def inked(self, force_n: float) -> bool:
        return float(force_n) > 0.0 and float(force_n) >= self.ink_threshold_n

    def ink_width(self, force_n: float) -> float:
        if not self.inked(force_n):
            return 0.0
        width = self.ink_width_min_m + self.ink_width_per_n_m * (float(force_n) - self.ink_threshold_n)
        return float(min(max(width, self.ink_width_min_m), self.ink_width_max_m))

    def compression_for(self, force_n: float) -> float:
        """Commanded compression for a target force, clamped to the holder's travel."""
        if force_n <= 0.0:
            return 0.0
        x = (float(force_n) - self.preload_n) / self.stiffness_n_per_m
        return float(min(max(x, 0.0), self.travel_m))


def force_profile(spec: Optional[Mapping], fraction: float, arc_m: float) -> Optional[float]:
    """Target force (N) at a point of a stroke, from a challenge's `pen_force:` mapping.

    fraction is the position along the stroke (0..1), arc_m the arc length drawn so far.
    profile: constant (target_n), ramp (start_n -> end_n), sine (mean_n +- amplitude_n with
    period_m along the stroke). None when there is no force target.
    """
    if not spec:
        return None
    kind = str(spec.get("profile", "constant")).lower()
    if kind == "constant":
        return float(spec["target_n"])
    if kind == "ramp":
        start, end = float(spec["start_n"]), float(spec["end_n"])
        return start + (end - start) * float(np.clip(fraction, 0.0, 1.0))
    if kind == "sine":
        mean, amplitude, period = float(spec["mean_n"]), float(spec["amplitude_n"]), float(spec["period_m"])
        if period <= 0:
            raise ValueError("pen_force.period_m must be positive")
        return mean + amplitude * math.sin(2.0 * math.pi * float(arc_m) / period)
    raise ValueError(f"Unknown pen_force profile {kind!r}; use constant, ramp or sine")
