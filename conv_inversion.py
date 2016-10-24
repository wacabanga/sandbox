"""Proof-of-concept inverse of tf.nn.conv2d"""

import tensorflow as tf


def filt_to_mat(filt, conv):
    # should be (n x m) x (n x m)
    pass


def inverse_conv2d(conv, filt):
    """
    The version which takes a filter (of type tf.Constant) as input.

    Assumes simple case: batch size 1, num filters 1.
    """
    conv_mat = filt_to_mat(filt, conv)
    input_dim = (filt.shape[0] + conv.shape[1] - 1,
                 filt.shape[1] + conv.shape[2] - 1)
    mult_op = tf.matmul(conv_mat,
                        tf.placeholder(conv.dtype,
                                       shape=input_dim[0] * input_dim[1]))
    # TODO compose with flatten
    return inverse_matmul(mult_op)  # should be from edgar's code
