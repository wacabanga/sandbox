"""Proof-of-concept inverse of tf.matmul in numpy"""

import numpy as np

def inverse_matmul(c):
    """
    Takes a full-rank matrix c as input, return inverse of matmul.
    Implemented in numpy as tensorflow doesn't have the necessary methods.
    """
    (n, k) = c.shape
    assert np.linalg.matrix_rank(c) == min(n, k)

    # create a rank x rank invertible random matrix
    # TODO need a more efficient way
    rank = min(n, k)
    while True:
        theta = np.random.rand(rank, rank)
        if np.linalg.det(theta) != 0:
            break

    max_dim = 2**5 # TODO decide on a limit on the larger dimension
    m = np.random.randint(rank, max_dim)
    print(m)
    # the matrix rows of c are independent
    if rank == k:
        for i in range(k, m):
            new_row = np.random.rand(1, k)
            theta = np.r_[theta, new_row]
        theta_left_inv = np.linalg.pinv(theta)
        a = np.matmul(c, theta_left_inv)
        b = theta
        return (a, b)

    # the matrix columns of c are independent
    if rank == n:
        for i in range(n, m):
            new_column = np.random.rand(n, 1)
            theta = np.r_[theta, new_column]
        theta_right_inv = np.linalg.pinv(theta)
        a = theta
        b = np.matmul(theta_right_inv, c)
        return (a, b)
