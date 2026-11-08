from __future__ import annotations


def projection_confidence_for_y(
    y_m: float,
    *,
    court_length_m: float = 16.0,
    near_confidence: float = 1.0,
    far_confidence: float = 0.65,
) -> float:
    """Depth-aware projection confidence for a camera behind the server.

    Confidence decreases toward the far baseline because the same pixel error
    represents a larger metric displacement under perspective compression.
    """
    if court_length_m <= 0:
        raise ValueError("court_length_m must be positive")
    if not 0.0 <= near_confidence <= 1.0 or not 0.0 <= far_confidence <= 1.0:
        raise ValueError("confidence bounds must be in [0, 1]")
    normalized = min(1.0, max(0.0, y_m / court_length_m))
    return near_confidence + (far_confidence - near_confidence) * normalized
