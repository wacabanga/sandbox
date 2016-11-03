"""Proof-of-concept inverse of 2D convolution"""

from typing import Tuple

import numpy as np
from scipy.fftpack import fftshift, ifftshift, fftn, ifftn
from scipy.signal import fftconvolve
import tensorflow as tf


def huge_matrix(filter_mat: np.ndarray,
                output_shape: Tuple[int, int]) -> np.ndarray:
    """
    This matrix contains a huge amount of redundant information and 0's.

    A specialized implementation could be optimized to not explicitly create
    this matrix.
    """
    (N, M) = filter_mat.shape
    (P, Q) = output_shape
    (R, S) = (N + P - 1, M + Q - 1)
    dim = R * S
    huge_mat = np.zeros((dim, dim))
    # obnoxious naive loop
    for i in range(P):
        for j in range(Q):
            for k in range(N):
                for l in range(M):
                    huge_mat[i * Q + j][(i + k) * S + (j + l)] = \
                        filter_mat[k][l]
    # we are in row echelon form! so we can just add a standard basis vector
    # for each pivot-less column:
    # http://math.stackexchange.com/questions/1530314/adding-linearly-independent-row-vectors-to-a-matrix
    row = P * Q
    for i in range(P):
        for j in range(Q, S):
            huge_mat[row][i * S + j] = 1.0
            row += 1
    for i in range(P, R):
        for j in range(S):
            huge_mat[row][i * S + j] = 1.0
            row += 1
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


def fwd_conv2d(input_mat, filter_mat):
    """Runs tf.nn.conv2d."""
    inp = tf.placeholder(tf.float32)
    filt = tf.placeholder(tf.float32)
    (N, M) = input_mat.shape
    (P, Q) = filter_mat.shape
    inp_feed = input_mat.reshape((1, N, M, 1))
    filt_feed = filter_mat.reshape((P, Q, 1, 1))
    op = tf.nn.conv2d(inp, filt, strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        return sess.run(op, feed_dict={inp: inp_feed, filt: filt_feed})


def fwd_conv2d_fft(input_mat, filter_mat):
    """Simulates tf.nn.conv2d using scipy fft."""
    flipped_mat = filter_mat[::-1, ::-1]
    # input_freq = fftshift(fftn(input_mat))
    # filter_freq = fftshift(fftn(flipped_mat))
    return fftconvolve(input_mat, flipped_mat, mode='same')
    # return fftshift(ifftn(ifftshift(input_freq * filter_freq)))


def test_fwd() -> None:
    """Tests fwd_conv2d vs fwd_conv2d_fft."""
    input_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    filter_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fwd = fwd_conv2d(input_mat, filter_mat)
    fwd_fft = fwd_conv2d_fft(input_mat, filter_mat)
    print(fwd)
    print(fwd_fft)


def test_inv() -> None:
    """Tests inverse conv2d on some random matrices."""
    ntests = 30
    for _ in range(ntests):
        # output_mat = np.array([[1, 2], [3, 4]])
        # filter_mat = np.array([[1, 2], [3, 4]])
        # theta = np.array([1, 2, 3, 4, 5])
        out_size = np.random.randint(1, 4, size=(2))
        filt_size = np.random.randint(1, 4, size=(2))
        theta_size = ((out_size[0] + filt_size[0] - 1)
                      * (out_size[1] + filt_size[1] - 1)
                      - out_size[0] * out_size[1])
        output_mat = np.random.rand(*out_size)
        filter_mat = np.random.rand(*filt_size)
        theta = np.random.randn(theta_size)
        input_mat = inverse_conv2d(output_mat, filter_mat, theta)
        convolved = fwd_conv2d(input_mat, filter_mat).reshape(out_size)
        error = np.linalg.norm(convolved - output_mat)
        assert error < 0.05, 'error %f too large!\n' \
                             'output:\n%s\ntheta:\n%s\n' \
                             'filt:\n%s\ninput:\n%s' % (error,
                                                        output_mat,
                                                        theta,
                                                        filter_mat,
                                                        input_mat)

if __name__ == '__main__':
    test_fwd()
    # test_inv()
