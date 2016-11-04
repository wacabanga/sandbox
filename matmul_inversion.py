"""Proof-of-concept inverse of tf.matmul in numpy"""

import numpy as np
from copy import deepcopy

def make_full_rank(matrix, lambda1):
    orig_matrix = deepcopy(matrix)
    (n, k) = matrix.shape
    r = min(n, k)
    if np.linalg.matrix_rank(matrix) == r:
        return matrix
    matrix[:r, :r] += np.identity(r)*lambda1
    assert np.linalg.matrix_rank(matrix) == r, "Didn't convert matrix to full-rank"
    return matrix

def inverse_matmul(output_mat, theta, lambda1=0.001):
    """
    Takes a n x k matrix output_mat as input, returns inverse of matmul.
    Parameter theta is of dimensions min(n, k) x m, for some m>=min(n, k).
    Implemented in numpy as tensorflow doesn't have the necessary methods.
    """
    (n, k) = output_mat.shape
    (rows, columns) = theta.shape
    assert (rows == min(n, k) and columns >= rows), "dimensions of theta not compatible"
    output_mat = make_full_rank(output_mat, lambda1)
    rank = min(n, k)

    # the matrix rows of c are independent
    if rank == k:
        theta = np.transpose(theta)      # theta becomes of dimensions m x k
        theta = make_full_rank(theta, lambda1)
        theta_left_inv = np.linalg.pinv(theta)
        a = np.matmul(output_mat, theta_left_inv)
        b = theta
        return (a, b)

    # the matrix columns of c are independent
    if rank == n:
        # theta is of dimensions n x m
        theta = make_full_rank(theta, lambda1)
        theta_right_inv = np.linalg.pinv(theta)
        a = theta
        b = np.matmul(theta_right_inv, output_mat)
        return (a, b)



def test_random():
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
        (inv_a, inv_b) = inverse_matmul(output_mat, theta)
        inv_output = np.matmul(inv_a, inv_b)

        error = np.linalg.norm(output_mat - inv_output)
        assert error < 1e-8, "error too large"

if __name__ == '__main__':
    test_random()
