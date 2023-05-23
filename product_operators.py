# product_operators.py
# Module for matrix product operators.

import numpy as np
from linalg_helpers import matrix_svd
import itertools

class ProductOperator:

    def __init__(self):

        self.tensors = []
        self.size = []


def tensor_to_mpo(tensor):
    """
    tensor_to_mpo(tensor)
    Converts the tensor representation of an operator into the equivalent MPO.
    The dimensions of the tensor should be ordered so that the "output state" dimensions
    come first, ordered from left to right, and then the "input states" come after, also
    ordered from left to right. If the Hamiltonian was expressed in terms of outer-product projectors,
    the first N dimensions of the tensor would correspond to the ket, and the last N dimensions to the bra
    of the outer product.

    Arugmens:
    tensor: np.array, tensor representation of the operator.

    Returns:
    mpo: ProductOperator, MPO decomposition of the tensor.
    """

    # Reshape the tensor so that the first two dimensions are the output and input 
    # for the first site, and so on until the last two dims are the output and input for the last.
    # Start by getting all of the relevant tensor shapes.
    old_shape = tensor.shape
    sites = len(old_shape) // 2
    assert len(old_shape) % 2 == 0, "Tensor must have an even number of axes."
    output_shape = old_shape[:sites]
    input_shape = old_shape[sites:]
    assert output_shape == input_shape, "Input and output dimensions must match."
    # Make a list of the new order for the axes.
    axis_permutation = list(itertools.chain.from_iterable(zip(range(sites), range(sites, 2 * sites))))
    # Reshape the tensor to interleave the input and output dimensions
    new_tensor = np.transpose(tensor, axes=axis_permutation)
    # Start by reshaping the tensor into 1xN, becuase each iteration we will reshape it into
    # (number of rows) * (output dimensions) * input dimension.
    phi = new_tensor.reshape((1, new_tensor.size))
    matrices = [] # List of the matrices to be shaped into tensors.
    for out_dim, in_dim in zip(output_shape[:-1], input_shape[:-1]):
        # Reshape the tensor
        new_rows = phi.shape[0] * out_dim * in_dim
        phi = phi.reshape((new_rows, phi.size // new_rows))
        # Perform the SVD
        u, s, vdag = matrix_svd(phi)
        # Append U to the list of matrices. Save S and V dagger for the next iteration.
        matrices.append(u)
        phi = s @ vdag
    matrices.append(phi) # The last site just gets the left-over matrix.
    # Now reshape the matrices into tensors.
    for i, (out_dim, in_dim) in enumerate(zip(output_shape, input_shape)):
        if i == 0:
            # For the left tensor, the sequence of SVD's leaves us a shape (out_dim, in_dim), a_1.
            # We want a final shape out_dim, in_dim, a_1.
            # NB: parentheses around indices mean they are grouped into the row or column of a matrix.
            matrices[i] = matrices[i].reshape((out_dim, in_dim, matrices[i].shape[1]))
        elif i == sites - 1:
            # For the rightmost tensor, the sequence of SVD's leaves a shape a_{L-1}, (out_dim, in_dim).
            # The desired final shape is a_{L-1}, out_dim, in_dim.
            matrices[i] = matrices[i].reshape((matrices[i].shape[0], out_dim, in_dim))
        else:
            # All others are middle tensors, with two bond legs and input and output legs.
            # After the sequence of SVD's, they have the shape (a_{i-1}, out_dim, in_dim), a_{i}.
            # We want a final shape of a_{i-1}, out_dim, in_dim, a_{i}.
            left_bond_dim = matrices[i].shape[0] // (out_dim * in_dim)
            matrices[i] = matrices[i].reshape((left_bond_dim, out_dim, in_dim, matrices[i].shape[1]))
    # Build an MPO with these tensors.
    mpo = ProductOperator()
    mpo.tensors = matrices
    mpo.size = len(matrices)
    return mpo