#!/usr/bin/env python3

"""Proof-of-concept tf.Graph inversion. Drafted in 2016-10-3 meeting."""

__author__ = "Lawrence Wu and Edgar Minasyan"
__status__ = "Prototype"

from queue import Queue
from typing import List, Tuple

from numpy.random import randn
import tensorflow as tf


def invert(output: tf.Tensor) -> List[tf.Tensor]:
    """Inverts a Tensorflow graph.

    Args:
        output (tf.Tensor): The output of the original graph.
    Returns:
        [tf.Tensor]. A list containing the outputs of the inverse graph.
    """
    # TODO accept a list of outputs
    # TODO handle Tensor reuse
    # TODO more ops
    # TODO handle not in domain
    # TODO make a useful API
    # TODO think about shapes other than scalars
    # naming: "new" means of inverse graph, "old" means of original graph
    new_name = output.name.split(':')[0]
    first_new_input = tf.placeholder(output.dtype, name=new_name)
    new_inputs = [first_new_input]
    new_outputs = []
    to_invert = Queue()
    to_invert.put((output, first_new_input))

    while not to_invert.empty():
        (old_tensor, new_tensor) = to_invert.get()
        old_op = old_tensor.op
        if old_op.type == 'Placeholder':
            new_name = 'rec_' + old_tensor.name
            new_name = new_name.split(':')[0]
            new_tensor = tf.identity(new_tensor, name=new_name)
            new_outputs.append(new_tensor)
        elif old_op.type == 'Identity':
            to_invert.put((old_op._inputs[0], new_tensor))
        elif old_op.type == 'Add':
            theta = tf.placeholder(old_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], new_tensor - theta))
            to_invert.put((old_op._inputs[1], theta))
        elif old_op.type == 'Mul':
            theta = tf.placeholder(old_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], new_tensor / theta))
            to_invert.put((old_op._inputs[1], theta))
        elif old_op.type == 'Div':
            theta = tf.placeholder(old_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], new_tensor * theta))
            to_invert.put((old_op._inputs[1], theta))
        elif old_op.type == 'Sub':
            theta = tf.placeholder(old_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], new_tensor + theta))
            to_invert.put((old_op._inputs[1], theta))
        elif old_op.type == 'Square':
            theta = tf.placeholder(tf.float32, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0],
                          tf.sqrt(new_tensor) * tf.sign(theta)))
        elif old_op.type == 'Sqrt':
            to_invert.put((old_op._inputs[0], tf.square(new_tensor)))
        elif old_op.type == 'Sign':
            theta = tf.placeholder(old_op._inputs[0].op.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0],
                          new_tensor * tf.abs(theta)))
        elif old_op.type == 'Maximum':
            theta = tf.placeholder(new_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0],
                          new_tensor + tf.minimum(0.0, theta)))
            to_invert.put((old_op._inputs[1],
                          new_tensor - tf.maximum(0.0, theta)))
        elif old_op.type == 'Minimum':
            theta = tf.placeholder(new_tensor.dtype, name='theta')
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0],
                          new_tensor - tf.minimum(0.0, theta)))
            to_invert.put((old_op._inputs[1],
                          new_tensor + tf.maximum(0.0, theta)))

    return new_outputs


def make_tests() -> List[Tuple[tf.Tensor, List[tf.Tensor]]]:
    """Generate test cases and visualization."""
    sess = tf.Session()
    tests = []

    a1 = tf.placeholder(tf.float32, name='a1')
    b1 = tf.placeholder(tf.float32, name='b1')
    i1 = tf.identity(a1 + b1, name='o1')
    o1 = invert(i1)
    tests.append((i1, o1))

    a2 = tf.placeholder(tf.float32, name='a2')
    b2 = tf.placeholder(tf.float32, name='b2')
    c2 = tf.placeholder(tf.float32, name='c2')
    i2 = tf.identity(a2 * b2 + c2, name='o2')
    o2 = invert(i2)
    tests.append((i2, o2))

    a3 = tf.placeholder(tf.float32, name='a3')
    b3 = tf.placeholder(tf.float32, name='b3')
    c3 = tf.placeholder(tf.float32, name='c3')
    i3 = tf.identity(tf.minimum(tf.maximum(a3, b3), c3), name='o3')
    o3 = invert(i3)
    tests.append((i3, o3))

    print("Generated (graph, inverse graph) pairs.")
    print("TODO: Test these.")

    TENSORBOARD_LOGDIR = "tensorboard_logdir"
    writer = tf.train.SummaryWriter(TENSORBOARD_LOGDIR, sess.graph)
    writer.flush()
    print("For graph visualization, invoke")
    print("$ tensorboard --logdir " + TENSORBOARD_LOGDIR)
    print("and click on the GRAPHS tab.")

    return tests

if __name__ == "__main__":
    make_tests()
