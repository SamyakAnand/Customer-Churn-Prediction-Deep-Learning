"""Validate the scalar probability returned by the churn model."""

import math


def normalize_probability(value):
    """Return a finite probability in [0, 1], or raise a safe ValueError."""
    if isinstance(value, bool):
        raise ValueError("The model output must be a numeric probability.")

    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The model output must be a numeric probability.") from exc

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("The model output must be finite and within [0, 1].")

    return probability


def extract_single_probability(prediction):
    """Require exactly one row with exactly one probability value."""
    try:
        rows = list(prediction)
    except (TypeError, ValueError) as exc:
        raise ValueError("The model must return a single probability value.") from exc

    if len(rows) != 1:
        raise ValueError("The model must return exactly one prediction row.")

    try:
        values = list(rows[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("The model output must have shape (1, 1).") from exc

    if len(values) != 1:
        raise ValueError("The model output must have shape (1, 1).")

    return normalize_probability(values[0])
