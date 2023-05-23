# random_states.py
# Generate random MPS's.

import numpy as np
import product_state as ps
import scipy.linalg as la

def random_state_vector(size):
    """
    random_state_vector(size
    
    Create a random vector of complex numbers
    that is normalized to serve as a random quantum state.

    Arguments:
    size: int, size of vecotr

    Returns:
    psi: np.array of complex, quantum state vector
    """

    # Generate vectors of real and imaginary parts,
    # each uniformly distributed in [-1, 1].
    # The add those together to make a complex vector.
    psi_real = 2.0 * np.random.rand(size) - 1.0
    psi_imag = 2.0 * np.random.rand(size) - 1.0
    psi = psi_real + 1j * psi_imag
    psi = psi / la.norm(psi)
    return psi


def random_mps(shape):
    """
    random_mps(shape)

    Create a random mps of a given shape.

    Arguments:
    shape: aray-like, shape of state tensor.

    Returns:
    mps: ProductState, random matrix product state.
    err: float, Error in the decomposition,
        norm of the decomposed tensor minus the original tensor.
    """

    total_size = np.prod(shape)
    psi = random_state_vector(total_size)
    state_tensor = psi.reshape(shape)
    mps = ps.tensor_to_mps(state_tensor)
    return mps

