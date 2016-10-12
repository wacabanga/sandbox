#!/usr/bin/env python3

"""Proof-of-concept tf.Graph inversion. Drafted in 2016-10-3 meeting."""

__author__ = "Lawrence Wu and Edgar Minasyan"
__status__ = "Prototype"

from queue import Queue
from typing import List, Tuple

from numpy.random import randn
import tensorflow as tf
import pdb


def invert(fwd_output: tf.Tensor): # Zen: 'fwd_output' less ambiguous than output
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
    # naming: "inv" means of inverse graph, "old" means of original graph
    # Zen: Maybe add 'inv' to the name
    inv_name = fwd_output.name.split(':')[0]
    first_inv_input = tf.placeholder(fwd_output.dtype, name=inv_name)
    inv_inputs = [first_inv_input]
    inv_outputs = []
    to_invert = Queue()
    # Queue of pairs of outputs of fwd_graph and corresponding input to inv?
    to_invert.put((fwd_output, first_inv_input))

    while not to_invert.empty():
        (fwd_tensor, inv_tensor) = to_invert.get()
        fwd_op = fwd_tensor.op
        if fwd_op.type == 'Placeholder':
            inv_name = 'rec_' + fwd_tensor.name # Zen: rec? reciprocal?
            inv_name = inv_name.split(':')[0]
            # Zen: Why create identity op and nost just return hte tensor?
            inv_tensor = tf.identity(inv_tensor, name=inv_name)
            inv_outputs.append(inv_tensor)
        elif fwd_op.type == 'Identity':
            to_invert.put((fwd_op._inputs[0], inv_tensor))
        elif fwd_op.type == 'Add':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0], inv_tensor - theta))
            to_invert.put((fwd_op._inputs[1], theta))
        elif fwd_op.type == 'Mul':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0], inv_tensor / theta))
            to_invert.put((fwd_op._inputs[1], theta))
        elif fwd_op.type == 'Div':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0], inv_tensor * theta))
            to_invert.put((fwd_op._inputs[1], theta))
        elif fwd_op.type == 'Sub':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0], inv_tensor + theta))
            to_invert.put((fwd_op._inputs[1], theta))
        elif fwd_op.type == 'Square':
            theta = tf.placeholder(tf.float32, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          tf.sqrt(inv_tensor) * tf.sign(theta)))
        elif fwd_op.type == 'Sqrt':
            to_invert.put((fwd_op._inputs[0], tf.square(inv_tensor)))
        elif fwd_op.type == 'Sign':
            theta = tf.placeholder(fwd_op._inputs[0].op.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor * tf.abs(theta)))
        elif fwd_op.type == 'Maximum':
            theta = tf.placeholder(inv_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor + tf.minimum(0.0, theta)))
            to_invert.put((fwd_op._inputs[1],
                          inv_tensor - tf.maximum(0.0, theta)))
        elif fwd_op.type == 'Minimum':
            theta = tf.placeholder(inv_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor - tf.minimum(0.0, theta)))
            to_invert.put((fwd_op._inputs[1],
                          inv_tensor + tf.maximum(0.0, theta)))
        # Function: x -> Ceiling[x]. Inverse Function: z -> z - theta, 0 <= theta < 1
        elif fwd_op.type == 'Ceil':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor - tf.sub(theta, tf.floor(theta))))
        # Function: x -> Floor[x]. Inverse Function: z -> z + theta, 0 <= theta < 1
        elif fwd_op.type == 'Floor':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor + tf.sub(theta, tf.floor(theta))))
        # Function: (x, y) -> x % y. Inverse Function: z -> (theta1 * theta2 + z, theta2), theta2 > z
        elif fwd_op.type == 'Mod':
            theta1 = tf.placeholder(fwd_tensor.dtype, name='theta1')
            theta2 = tf.placeholder(fwd_tensor.dtype, name='theta2')
            inv_inputs.append(theta1)
            inv_inputs.append(theta2)
            # make theta2 be smaller larger than inv_tensor
            to_invert.put((fwd_op._inputs[0],
                          theta1 * theta2 + inv_tensor))
            to_invert.put((fwd_op._inputs[1],
                          theta2))
        # Function: x -> |x|. Inverse Function: z -> theta * z, theta from {-1, 1}
        elif fwd_op.type == 'Abs':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          inv_tensor * tf.sign*(theta)))
        # Function: (x, y) -> x^y. Inverse Function: z -> (theta, log_{theta}(z))
        elif fwd_op.type == 'Pow':
            theta = tf.placeholder(fwd_tensor.dtype, name='theta')
            inv_inputs.append(theta)
            to_invert.put((fwd_op._inputs[0],
                          theta))
            to_invert.put((fwd_op._inputs[1],
                          tf.div(tf.log(inv_tensor), tf.log(theta))))

    return inv_outputs


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

    a4 = tf.placeholder(tf.float32, name='a4')
    b4 = tf.placeholder(tf.float32, name='b4')
    c4 = tf.placeholder(tf.float32, name='c4')
    i4 = tf.identity(tf.ceil(tf.mod(tf.pow(a4, b4), c4)), name='o4')
    o4 = invert(i4)
    tests.append((i4, o4))

    # Zen: This will result in a graph with three outputs.
    #      Make sure your solution also makes the following work
    a5 = tf.placeholder(tf.float32, name='a5')
    b5 = tf.placeholder(tf.float32, name='b5')
    i5 = tf.identity(a5 * b5 + a5, name='o5')
    o5 = invert(i5)
    tests.append((i5, o5))

    a6 = tf.placeholder(tf.float32, name='a6')
    b6 = tf.placeholder(tf.float32, name='b6')
    i6 = tf.identity(b6 ** (a6 * b6 + a6), name='o6')
    o6 = invert(i6)
    tests.append((i6, o6))

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
