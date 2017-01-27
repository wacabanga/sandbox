"""Proof-of-concept inverse of 2D convolution"""

from typing import Tuple

import numpy as np
from scipy import fftpack
from scipy.signal import fftconvolve
import tensorflow as tf


def fwd_conv2d(input_mat: np.ndarray, filter_mat: np.ndarray) -> np.ndarray:
    """Runs tf.nn.conv2d."""
    inp = tf.placeholder(tf.float32)
    filt = tf.placeholder(tf.float32)
    (N, M) = input_mat.shape
    (P, Q) = filter_mat.shape
    inp_feed = input_mat.reshape((1, N, M, 1))
    filt_feed = filter_mat.reshape((P, Q, 1, 1))
    op = tf.nn.conv2d(inp, filt, strides=[1, 1, 1, 1], padding='VALID')
    with tf.Session() as sess:
        return sess.run(op, feed_dict={inp: inp_feed, filt: filt_feed})


def _huge(filter_mat: np.ndarray,
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


def inv_conv2d(output_mat: np.ndarray,
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
    huge_mat = _huge(filter_mat, (P, Q))
    flattened = np.concatenate((output_mat.reshape(P * Q), theta))
    input_flattened = np.dot(np.linalg.inv(huge_mat), flattened)
    return input_flattened.reshape((N + P - 1, M + Q - 1))


def fwd_conv2d_fft(input_mat: np.ndarray,
                   filter_mat: np.ndarray) -> np.ndarray:
    """Simulates tf.nn.conv2d using scipy fft."""
    flipped_mat = filter_mat[::-1, ::-1]
    return fftconvolve(input_mat, flipped_mat, mode='full')


def _uncentered(arr: np.ndarray, newshape: Tuple[int, ...]) -> np.ndarray:
    """See scipy/signal/signaltools.py:_centered."""
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (newshape - currshape) // 2
    endind = startind + currshape
    myslice = [(startind[k],
                newshape[k] - endind[k]) for k in range(len(endind))]
    return np.lib.pad(arr, tuple(myslice), 'constant')


def inv_conv2d_fft(output_mat: np.ndarray,
                   filter_mat: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
    """Parametric inverse implemented with FFT (faster)."""
    output_shape = np.asarray(output_mat.shape)
    filter_shape = np.asarray(filter_mat.shape)
    input_shape = output_shape - filter_shape + 1

    fshape = [fftpack.helper.next_fast_len(int(d)) for d in output_shape]
    fslice = tuple([slice(0, int(sz)) for sz in input_shape])

    output_mat = _uncentered(output_mat, fshape)
    flipped_mat = filter_mat[::-1, ::-1]

    sp1 = np.fft.rfftn(output_mat, fshape)
    sp2 = np.fft.rfftn(flipped_mat, fshape)
    ret = np.fft.irfftn(sp1 / sp2, fshape)[fslice].copy()
    return ret.real


def test_inv_fft() -> None:
    """Tests FFT-based inverse conv2d on some random matrices"""
    ntests = 30
    for i in range(ntests):
        input_size = np.random.randint(4, 16, size=(2))
        filter_size = np.random.randint(1, 4, size=(2))
        input_mat = np.random.rand(*input_size)
        filter_mat = np.random.rand(*filter_size)
        # input_mat = np.array([[1, 2], [3, 4]])
        # filter_mat = np.array([[1, 2], [3, 4]])
        # fwd = fwd_conv2d(input_mat, filter_mat)
        fwd_fft = fwd_conv2d_fft(input_mat, filter_mat)
        inv_fwd_fft = inv_conv2d_fft(fwd_fft, filter_mat, None)
        fwd_inv_fwd_fft = fwd_conv2d_fft(inv_fwd_fft, filter_mat)
        # print(input_mat)
        # print(filter_mat)
        # print(fwd)
        # print(fwd_fft)
        # print(inv_fwd_fft)
        # print(fwd_inv_fwd_fft)
        error = np.linalg.norm(fwd_inv_fwd_fft - fwd_fft)
        assert error < 0.05, 'test_inv_fft test #%d failed:\n' \
                             'error %f too large!\n' \
                             'filter_mat:\n%s\ninput_mat:\n%s\n' \
                             'fwd_fft:\n%s\n' \
                             'inv_fwd_fft:\n%s\n' % (i,
                                                     error,
                                                     filter_mat,
                                                     input_mat,
                                                     fwd_fft,
                                                     inv_fwd_fft)


def test_inv() -> None:
    """Tests non-FFT inverse conv2d on some random matrices."""
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
        input_mat = inv_conv2d(output_mat, filter_mat, theta)
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
    test_inv()
    test_inv_fft()
