"""Small dependency-free shape-preserving cubic Hermite interpolator."""
from __future__ import annotations

from bisect import bisect_right
from typing import Sequence


def pchip_slopes(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Return Fritsch-Carlson/PCHIP nodal first derivatives."""
    count = len(x)
    if count != len(y) or count < 2:
        raise ValueError("PCHIP requires matching arrays with at least two points")
    intervals = [x[index + 1] - x[index] for index in range(count - 1)]
    if any(value <= 0 for value in intervals):
        raise ValueError("PCHIP x coordinates must strictly increase")
    secants = [
        (y[index + 1] - y[index]) / intervals[index]
        for index in range(count - 1)
    ]
    if count == 2:
        return [secants[0], secants[0]]

    slopes = [0.0] * count
    for index in range(1, count - 1):
        left, right = secants[index - 1], secants[index]
        if left == 0 or right == 0 or left * right < 0:
            slopes[index] = 0.0
        else:
            weight_left = 2 * intervals[index] + intervals[index - 1]
            weight_right = intervals[index] + 2 * intervals[index - 1]
            slopes[index] = (
                (weight_left + weight_right)
                / (weight_left / left + weight_right / right)
            )

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if value * d0 <= 0:
            return 0.0
        if d0 * d1 < 0 and abs(value) > abs(3 * d0):
            return 3 * d0
        return value

    slopes[0] = endpoint(intervals[0], intervals[1], secants[0], secants[1])
    slopes[-1] = endpoint(
        intervals[-1], intervals[-2], secants[-1], secants[-2])
    return slopes


def pchip_interpolate(x: Sequence[float], y: Sequence[float], value: float) -> float:
    """Evaluate PCHIP inside the source domain; extrapolation is forbidden."""
    if value < x[0] or value > x[-1]:
        raise ValueError("PCHIP extrapolation is forbidden")
    if value == x[-1]:
        return float(y[-1])
    right = bisect_right(x, value)
    left = right - 1
    if value == x[left]:
        return float(y[left])
    slopes = pchip_slopes(x, y)
    width = x[right] - x[left]
    fraction = (value - x[left]) / width
    h00 = 2 * fraction**3 - 3 * fraction**2 + 1
    h10 = fraction**3 - 2 * fraction**2 + fraction
    h01 = -2 * fraction**3 + 3 * fraction**2
    h11 = fraction**3 - fraction**2
    return (
        h00 * y[left] + h10 * width * slopes[left]
        + h01 * y[right] + h11 * width * slopes[right]
    )


def pchip_derivative(x: Sequence[float], y: Sequence[float], value: float) -> float:
    """Evaluate the first derivative of the same PCHIP interpolant."""
    if value < x[0] or value > x[-1]:
        raise ValueError("PCHIP extrapolation is forbidden")
    slopes = pchip_slopes(x, y)
    if value == x[0]:
        return slopes[0]
    if value == x[-1]:
        return slopes[-1]
    right = bisect_right(x, value)
    if value == x[right - 1]:
        return slopes[right - 1]
    left = right - 1
    width = x[right] - x[left]
    fraction = (value - x[left]) / width
    return (
        (6 * fraction**2 - 6 * fraction) * y[left] / width
        + (3 * fraction**2 - 4 * fraction + 1) * slopes[left]
        + (-6 * fraction**2 + 6 * fraction) * y[right] / width
        + (3 * fraction**2 - 2 * fraction) * slopes[right]
    )
