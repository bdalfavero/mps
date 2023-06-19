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

def main():
    psi_mps = random_mps((2,) * 3)
    print("MPS shapes:")
    for tensor in psi_mps.get_tensor_list():
        print(tensor.shape)
    inner_product = ps.mps_inner_product(psi_mps, psi_mps)
    print("MPS self inner product: ", inner_product)
    id2 = np.eye(2, dtype="complex")
    heisenberg2 = hb.heisenberg_two_spin_hamiltonian(np.ones(3))
    three_spin_hamiltonian = np.kron(heisenberg2, id2) + np.kron(id2, heisenberg2) 
    hamiltonian_tensor = three_spin_hamiltonian.reshape((2,) * 6)
    hamiltonian_mpo = po.tensor_to_mpo(hamiltonian_tensor)
    print("MPO shapes:")
    for tensor in hamiltonian_mpo.tensors:
        print(tensor.shape)


if __name__ == "__main__":
    main()
