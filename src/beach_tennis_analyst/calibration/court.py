from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BeachTennisCourt:
    """Metric court model using the doubles dimensions for beach tennis."""

    width_m: float = 8.0
    length_m: float = 16.0
    net_y_m: float = 8.0

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.length_m <= 0:
            raise ValueError("Court dimensions must be positive")
        if not 0 < self.net_y_m < self.length_m:
            raise ValueError("Net coordinate must lie inside the court")

    @property
    def metric_corners(self) -> np.ndarray:
        """Corners ordered: near-left, near-right, far-right, far-left."""
        return np.asarray(
            [
                [0.0, 0.0],
                [self.width_m, 0.0],
                [self.width_m, self.length_m],
                [0.0, self.length_m],
            ],
            dtype=np.float32,
        )

    def contains(self, x_m: float, y_m: float, margin_m: float = 0.0) -> bool:
        return (
            -margin_m <= x_m <= self.width_m + margin_m
            and -margin_m <= y_m <= self.length_m + margin_m
        )

    def side_for_y(self, y_m: float) -> str:
        return "near" if y_m < self.net_y_m else "far"
