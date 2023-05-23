import numpy as np

# Define sigma matrices for SU(2), including the identity.
sigma = {}
sigma["id"] = np.eye(2, dtype="complex")
sigma["x"] = np.array([[0., 1.], [1., 0.]], dtype="complex")
sigma["y"] = np.array([[0., -1.0j], [1.0j, 0.]], dtype="complex")
sigma["z"] = np.array([[1., 0.], [0., -1.]], dtype="complex")

def heisenberg_two_spin_hamiltonian(j):
    """
    heisenberg_two_spin_hamiltonian(j)
    Returns a 2-spin Heisenberg Hamiltonian with coefficients j.

    Arguments:
    j: np.array, vector of three coefficients.

    Returns:
    hamiltonian: np.array, a 4x4 matrix for the 2-spin Hamiltonian.
    """

    ham = j[0] * np.kron(sigma["x"], sigma["x"]) \
        + j[1] * np.kron(sigma["y"], sigma["y"]) \
        + j[2] * np.kron(sigma["z"], sigma["z"])
    return ham

def heisenberg_two_spin_tensor(j):
    """
    heisenberg_two_spin_tensor(j)
    Returns a tensor representation of the 2-spin Heisenberg hamiltonian with coefficients j.
    NB if the Hamiltonian is represented by
    $\hat{H} = \sum_{\sigma} \sum_{\tau} h^{\sigma_1 \sigma_2}_{\tau_1 \tau_2}
    \ket{\sigma_1 \sigma_2} \bra{\tau_1 \tau_2}$,
    then the tensor will have the indices in order $\sigma_1, \sigma_2, \tau_1, \tau_2$.

    Arguments:
    j: np.array, coefficients for Hamiltonian. Passed to heisenberg_two_spin_hamiltonian().

    Returns:
    hamiltonian: np.array, 2x2x2x2 tensor of Hamiltonian.
    """
    hamiltonian = heisenberg_two_spin_hamiltonian(j)
    return hamiltonian.reshape((2, 2, 2, 2))