from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from beach_tennis_analyst.domain.models import ObservationStatus, PlayerFrame


@dataclass(frozen=True, slots=True)
class GridCellOccupancy:
    row: int
    column: int
    samples: int
    ratio: float


def occupancy_grid(
    frames: Iterable[PlayerFrame],
    *,
    court_width_m: float = 8.0,
    court_length_m: float = 16.0,
    columns: int = 3,
    rows: int = 3,
) -> list[GridCellOccupancy]:
    valid = [
        frame
        for frame in frames
        if frame.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
        and 0.0 <= frame.x_m <= court_width_m
        and 0.0 <= frame.y_m <= court_length_m
    ]
    counts = np.zeros((rows, columns), dtype=np.int64)
    for frame in valid:
        column = min(columns - 1, int(frame.x_m / court_width_m * columns))
        row = min(rows - 1, int(frame.y_m / court_length_m * rows))
        counts[row, column] += 1

    total = int(counts.sum())
    return [
        GridCellOccupancy(
            row=row,
            column=column,
            samples=int(counts[row, column]),
            ratio=0.0 if total == 0 else float(counts[row, column] / total),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def heatmap_matrix(
    frames: Iterable[PlayerFrame],
    *,
    court_width_m: float = 8.0,
    court_length_m: float = 16.0,
    bins_x: int = 40,
    bins_y: int = 80,
) -> np.ndarray:
    points = [
        (frame.x_m, frame.y_m)
        for frame in frames
        if frame.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
        and 0.0 <= frame.x_m <= court_width_m
        and 0.0 <= frame.y_m <= court_length_m
    ]
    if not points:
        return np.zeros((bins_y, bins_x), dtype=np.float32)
    x = np.asarray([point[0] for point in points], dtype=np.float32)
    y = np.asarray([point[1] for point in points], dtype=np.float32)
    matrix, _, _ = np.histogram2d(
        y,
        x,
        bins=(bins_y, bins_x),
        range=((0.0, court_length_m), (0.0, court_width_m)),
    )
    if matrix.max() > 0:
        matrix = matrix / matrix.max()
    return matrix.astype(np.float32)
