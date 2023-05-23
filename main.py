#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la
from linalg_helpers import matrix_svd
import product_state as ps
import product_operators as po
from random_states import random_mps
from copy import copy, deepcopy
from ncon import ncon
import heisenberg as hb

def test_left_normalization(mps):
    # Test that the state is left-normalized.

    # Reshape the first and last tensors for ease.
    left_old_shape = mps.tensors[0].shape
    right_old_shape = mps.tensors[-1].shape
    mps.tensors[0] = mps.tensors[0].reshape(
        (1, mps.tensors[0].shape[0], mps.tensors[0].shape[1])
    )
    mps.tensors[-1] = mps.tensors[-1].reshape(
        (mps.tensors[-1].shape[0], mps.tensors[-1].shape[1], 1)
    )
    # Compute \sum_\sigma A^\dagger^\sigma A^\sigma. It should be the identity.
    for op in mps.tensors:
        # Reshape the tensor into a matrix.
        u = op.reshape((op.size // op.shape[-1], op.shape[-1]))
        # Compute U^\dagger U for that matrix.
        u_dag_u = u.conj().T @ u
        print(la.norm(u_dag_u - np.eye(u_dag_u.shape[0])), np.sum(np.diag(u_dag_u) - 1.0))
    # Change the shapes back.
    mps.tensors[0] = mps.tensors[0].reshape(left_old_shape)
    mps.tensors[-1] = mps.tensors[-1].reshape(right_old_shape)

def main():
    id2 = np.eye(2, dtype="complex")
    heisenberg2 = hb.heisenberg_two_spin_hamiltonian(np.ones(3))
    three_spin_hamiltonian = np.kron(heisenberg2, id2) + np.kron(id2, heisenberg2) 
    hamiltonian_tensor = three_spin_hamiltonian.reshape((2,) * 6)
    hamiltonian_mpo = po.tensor_to_mpo(hamiltonian_tensor)
    for tensor in hamiltonian_mpo.tensors:
        print(tensor.shape)

if __name__ == "__main__":
    main()
