"""Proof-of-concept inverse of 2D convolution"""

from typing import Tuple

import numpy as np


def huge_matrix(filter_mat: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """
    This matrix contains a huge amount of redundant information and 0's.

    A specialized implementation could be optimized to not explicitly create
    this matrix.
    """
    (N, M) = filter_mat.shape
    (P, Q) = output_shape
    dim = (N + P - 1) * (M + Q - 1)
    huge_mat = np.zeros(dim, dim)
    # obnoxious naive loop
    for i in range(P):
        for j in range(Q):
            for k in range(N):
                for l in range(M):
                    huge_mat[i * Q + j][(i + j) * Q + (j + l)] = filter_mat[k][l]
    for i in range(P * Q, dim):
        # TODO: pick a choice which makes huge_mat invertible
        huge_mat[i][i] = 1.0
    return huge_mat


def inverse_conv2d(output_mat: np.ndarray,
                   filter_mat: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
    """
    Given g and h, finds f such that f * g = h.

    The dimension of theta is the number of elements in f
    minus the number of elements in h.
    """
    (N, M) = filter_mat.shape
    (P, Q) = output_mat.shape
    dim = (N + P - 1) * (M + Q - 1)
    assert len(theta) == dim - P * Q, 'theta dimensions wrong'
    huge_mat = huge_matrix(filter_mat, (P, Q))
    flattened = np.concatenate((output_mat.reshape(P * Q), theta))
    input_flattened = np.dot(np.linalg.inv(huge_mat), flattened)
    return input_flattened.reshape((N + P - 1, M + Q - 1))


def test_random() -> None:
    """Tests some random 2x2 matrices."""
    output_mat = np.random.randn((2, 2))
    filter_mat = np.random.randn((2, 2))
    theta = np.random.randn(5)
    print(output_mat)
    print(filter_mat)
    print(theta)
    print(inverse_conv2d(output_mat, filter_mat, theta))

if __name__ == '__main__':
    test_random()
