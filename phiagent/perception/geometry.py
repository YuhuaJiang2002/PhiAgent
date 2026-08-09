"""Small dependency-free rotation utilities shared by teacher adapters."""

from __future__ import annotations

import math
from typing import Iterable

from phiagent.physical_language.schema import QuaternionXYZW


def rotation_matrix_to_quaternion(
    values: Iterable[Iterable[float]],
) -> QuaternionXYZW:
    """Convert a proper 3x3 rotation matrix to a canonical XYZW quaternion."""

    matrix = tuple(tuple(float(value) for value in row) for row in values)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("rotation matrix must have shape 3x3")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("rotation matrix must contain only finite values")
    for left in range(3):
        for right in range(3):
            dot = sum(matrix[index][left] * matrix[index][right] for index in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1e-4:
                raise ValueError("rotation matrix is not orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError("rotation matrix must have determinant +1")
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)
