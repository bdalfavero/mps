# product_operators.py
# Module for matrix product operators.

import numpy as np
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
    old_shape = tensor.shape
    sites = len(old_shape) // 2
    assert len(old_shape) % 2 == 0, "Tensor must have an even number of axes."
    output_shape = old_shape[:sites]
    input_shape = old_shape[sites:]
    assert output_shape == input_shape, "Input and output dimensions must match."
    # Make a list of the new order for the axes.
    axis_permutation = list(itertools.chain.from_iterable(zip(range(sites), range(sites, 2 * sites))))
    new_tensor = np.transpose(tensor, axes=axis_permutation)
    op_matrix = 
    mpo = ProductOperator()
    return mpo