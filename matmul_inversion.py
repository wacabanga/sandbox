"""Proof-of-concept inverse of tf.matmul in numpy and tensorflow"""

import numpy as np
import tensorflow as tf

def np_make_full_rank(matrix, lambda1):
    (n, k) = matrix.shape
    r = min(n, k)
    if np.linalg.matrix_rank(matrix) == r:
        return matrix
    matrix[:r, :r] += np.identity(r)*lambda1
    assert np.linalg.matrix_rank(matrix) == r, "Didn't convert matrix to full-rank"
    return matrix

def np_inverse_matmul(output_mat, theta, lambda1=0.001):
    """
    Takes a n x k matrix output_mat as input, returns inverse of matmul.
    Parameter theta is of dimensions min(n, k) x m, for some m>=min(n, k).
    Implemented in numpy.
    """
    (n, k) = output_mat.shape
    (rows, columns) = theta.shape
    assert (rows == min(n, k) and columns >= rows), "dimensions of theta not compatible"
    output_mat = np_make_full_rank(output_mat, lambda1)
    rank = min(n, k)

    # the matrix rows of c are independent
    if rank == k:
        theta = np.transpose(theta)      # theta becomes of dimensions m x k
        theta = np_make_full_rank(theta, lambda1)
        theta_left_inv = np.linalg.pinv(theta)
        a = np.matmul(output_mat, theta_left_inv)
        b = theta
        return (a, b)

    # the matrix columns of c are independent
    if rank == n:
        # theta is of dimensions n x m
        theta = np_make_full_rank(theta, lambda1)
        theta_right_inv = np.linalg.pinv(theta)
        a = theta
        b = np.matmul(theta_right_inv, output_mat)
        return (a, b)

def np_test_random():
    """Tests some random matrices"""
    NTESTS = 30
    for _ in range(NTESTS):
        # get random dimensions
        input_a_rows = np.random.randint(1, 10)
        input_a_columns = np.random.randint(1, 10)
        input_b_rows = input_a_columns
        input_b_columns = np.random.randint(1, 10)
        theta_row_size = min(input_a_rows, input_b_columns)
        theta_col_size = theta_row_size + np.random.randint(0, 10)

        # create the random matrices and parameters
        input_a = np.random.rand(input_a_rows, input_a_columns)
        input_b = np.random.rand(input_b_rows, input_b_columns)
        theta = np.random.rand(theta_row_size, theta_col_size)

        output_mat = np.matmul(input_a, input_b)
        (inv_a, inv_b) = np_inverse_matmul(output_mat, theta)
        inv_output = np.matmul(inv_a, inv_b)

        error = np.linalg.norm(output_mat - inv_output)
        assert error < 1e-8, "error too large"


def tf_make_full_rank(matrix, lambda1):
    """
    Makes the given n x k matrix M full-rank the following way:
    M = M + lambda * I, if M is not full-rank and I is the identity matrix of size min(n, k).
    """
    (n, k) = (matrix.get_shape().as_list()[0], matrix.get_shape().as_list()[1])
    r = min(n, k)
    # Add lambda * I to the corresponding part of the given matrix.
    sub1 = matrix[:r, :r] + tf.scalar_mul(lambda1, tf.eye(r))

    # Construct the resulting matrix accordingly.
    if r == n:
        out_matrix = tf.concat(1, [sub1, matrix[:, r:]])
    else:
        out_matrix = tf.concat(0, [sub1, matrix[r:, :]])

    # Choose the constructed matrix only if the given one is not full-rank.
    condition = tf.not_equal(tf.matrix_determinant(matrix[:r, :r]), 0.0)
    result = tf.cond(condition, lambda: tf.identity(matrix), lambda: tf.identity(out_matrix))

    with tf.control_dependencies([tf.Assert(tf.not_equal(tf.matrix_determinant(result[:r, :r]), 0.0), [result])]):
        return result

def tf_pseudoinverse(matrix):
    """
    Computes the Moore-Penrose pseudoinverse of the full-rank matrix
    M in tensorflow using SVD decomposition the following way.
    S, U, V = SVD(M), where S, U, V are the matrices U * S * transpose(V) = M.

    In order to obtain the Moore-Penrose pseudoinverse, use the following formula:
    pinv(M) = V * pinv(S) * transpose(U), where pinv(S) is calculated by taking
    the reciprocal of each non-zero element (on the diagonal) and transposing.
    """
    # Get the SVD decomposition of the given matrix
    s, u, v = tf.svd(matrix, full_matrices=True, compute_uv=True)
    (n, k) = (matrix.get_shape().as_list()[0], matrix.get_shape().as_list()[1])

    # Make s a matrix, reciprocate the singular values and transpose to get pinv(s).
    s_plus = tf.diag(tf.reciprocal(s))
    if n <= k:
        s_plus = tf.concat(1, [s_plus, tf.zeros([n, k-n], tf.float32)])
    else:
        s_plus = tf.concat(0, [s_plus, tf.zeros([n-k, k], tf.float32)])
    s_plus = tf.transpose(s_plus)

    # Obtain the pseudoinverse of the given matrix with the formula mentioned above.
    pinv = tf.matmul(v, tf.matmul(s_plus, tf.transpose(u)))
    return pinv

def tf_inverse_matmul(output_mat, theta, lambda1=1e-6):
    """
    Takes a n x k matrix output_mat as input, returns inverse of matmul.
    Parameter theta is of dimensions min(n, k) x m, for some m>=min(n, k).
    The inverse is a tuple (A, B) of matrices s.t. A * B equals output_mat.
    Two cases depending on (n, k):
        (i) A = C * Theta^(-1), B = Theta
        (ii) A = Theta, B = Theta^(-1) * C
    """
    (n, k) = (output_mat.get_shape().as_list()[0], output_mat.get_shape().as_list()[1])
    (rows, columns) = (theta.get_shape().as_list()[0], theta.get_shape().as_list()[1])
    assert (rows == min(n, k) and columns >= rows), "dimensions of theta not compatible"
    # make the given matrix full-rank.
    output_mat = tf_make_full_rank(output_mat, lambda1)

    # the matrix rows of output_mat are independent
    if k <= n:
        # Case 1: C = (C * Theta^(-1)) * Theta
        theta = tf.matrix_transpose(theta)
        theta = tf_make_full_rank(theta, lambda1)
        theta_left_inv = tf_pseudoinverse(theta)
        a = tf.matmul(output_mat, theta_left_inv)
        b = theta
        return (a, b)

    # the matrix columns of output_mat are independent
    if n < k:
        # Case 2: C = Theta * (Theta^(-1) * C)
        theta = tf_make_full_rank(theta, lambda1)
        theta_right_inv = tf_pseudoinverse(theta)
        a = theta
        b = tf.matmul(theta_right_inv, output_mat)
        return (a, b)

def tf_test_random():
    """Tests some random matrices"""
    NTESTS = 30
    for _ in range(NTESTS):
        # Get random dimensions.
        input_a_rows = np.random.randint(1, 10)
        input_a_columns = np.random.randint(1, 10)
        input_b_rows = input_a_columns
        input_b_columns = np.random.randint(1, 10)
        theta_row_size = min(input_a_rows, input_b_columns)
        theta_col_size = theta_row_size + np.random.randint(0, 10)

        # Create the random test matrices and the parameter matrix.
        input_a = np.random.rand(input_a_rows, input_a_columns)
        input_b = np.random.rand(input_b_rows, input_b_columns)
        theta = np.random.rand(theta_row_size, theta_col_size)
        output = np.matmul(input_a, input_b)

        # Run the inversion in tensorflow.
        output_tf = tf.placeholder(tf.float32, shape=(input_a_rows, input_b_columns))
        theta_tf = tf.placeholder(tf.float32, shape=(theta_row_size, theta_col_size))
        (inv_a, inv_b) = tf_inverse_matmul(output_tf, theta_tf)
        inv_output = tf.matmul(inv_a, inv_b)

        # Get the error between the given matrix and the matrix obtained after inversion.
        # Assert that the error is negligible.
        error = tf.reduce_sum(tf.square(tf.sub(output_tf, inv_output))) / (input_a_rows * input_b_columns)
        with tf.control_dependencies([tf.assert_less(error, 1e-8)]):
            error = tf.Print(error, [error])
        with tf.Session() as sess:
            sess.run(error, feed_dict={output_tf: output, theta_tf: theta})

if __name__ == '__main__':
    tf_test_random()
