# product_state.py
# Class definition and helper functions for matrix product states

import numpy as np
import scipy.linalg as la
from linalg_helpers import matrix_svd
from ncon import ncon

class ProductState:

    def __init__(self):
        self.tensors = []
        self.size = 0

    def to_tensor(self):
        """
        mps_to_tensor(mps)

        Contract the MPS along the bonds to create a tensor.

        Arguments:
        None.

        Returns:
        tensor: np.array, tensor version of the mps.
        """

        new_tensor = self.tensors[0]
        for i in range(1, len(self.tensors)):
            new_tensor = np.tensordot(new_tensor, self.tensors[i], axes=(-1, 0))
        return new_tensor


def tensor_to_mps(tensor):
    """
    tensor_to_mps(tensor)

    Convert a tensor to a tensor train singular value decomposition.
    This function does not compress the MPS. The matrix product state
    is left canonical.

    Arguments:
    tensor: np.array, tensor for quantum state. The size of each dimension
    is the dimension of each site's Hilbert space. The dimensions go from the
    left-most site to the right-most on the last dimension.

    Returns:
    mps object
    """

    # Store the old shape before doing operations on the tensor.
    # The size of each dimension in the tensor is the dimension of a site's
    # Hilbert space.
    old_shape = tensor.shape
    assert len(old_shape) >= 2, "The tensor must have 2 or more dimensions."
    # Reshape the tensor into a matrix where the number of rows is
    # the dimension of the first site's Hilbert space.
    psi = tensor.reshape((1, tensor.size))
    # Loop through the site dimensions and do the SVD. Store the matrices in a list.
    matrices = []
    for dim in old_shape[:-1]:
        # Reshape the tensor so that the number of rows is the dimension
        # of the current site. 
        psi = psi.reshape((dim * psi.shape[0], psi.shape[1] // dim))
        u, s, v_dag = matrix_svd(psi)
        matrices.append(u)
        psi = s @ v_dag
    matrices.append(psi)
    # Reshape the matrices into tensors.
    for i in range(len(matrices)):
        if (i != 0) and (i != len(matrices) - 1):
            new_shape = (matrices[i - 1].shape[-1], old_shape[i], \
                matrices[i].size // (matrices[i - 1].shape[-1] * old_shape[i]))
            matrices[i] = matrices[i].reshape(new_shape)
    matrices[-1] = matrices[-1].reshape((matrices[-1].size // old_shape[-1], old_shape[-1]))
    # Create an MPS object from the matrices.
    mps = ProductState()
    mps.tensors = matrices
    mps.size = len(matrices)
    return mps


def mps_inner_product(mps1, mps2):
    """
    mps_inner_product(mps1, mps2)
    Take the inner product of two MPS. See Scholwoeck Fig. 21.

    Arguments:
    mps1: ProductState, first mps in the inner product. This one will have its h.c. taken.
    mps2: ProductState, second mps. It does not have the h.c. taken.

    Returns:
    inner_product: Float, inner product of the two MPS.
    """
    
    # NCON requires two lists: a list of tensors to contract, and a list of contractions.
    # We will contract the conjugate of the tensors for the first state, and the normal tensors
    # for the second.
    tensor_list = [t.conj() for t in mps1.tensors] + mps2.tensors
    # Make a list of lists to store the contraction indices.
    v = []
    for t in tensor_list:
        v.append([0] * len(t.shape))
    # We also need to contract in the order given on Schollwoeck Fig. 21
    # Iterate through the pairs of tensors, assigning contraction numbers as we go.
    contraction_ix = 0 # Label for next contraction, incrememnted after each assignment.
    for ix in range(len(mps1.tensors)):
        upper_ix = ix # Index within tensor_list for current tensor from mps1
        lower_ix = ix + len(mps2.tensors) # Index within tensor_list for current tensor from mps2
        if ix == 0:
            # Assign contractions for the leftmost tensors.
            # Contract the vertical legs on the upper and lower tensors,
            # then contract the horizontal legs on the upper tensor and its neighbor,
            # then do the same for the lower tensor.
            v[upper_ix][0] = contraction_ix
            v[lower_ix][0] = contraction_ix
            contraction_ix += 1
            v[upper_ix][1] = contraction_ix
            v[upper_ix + 1][0] = contraction_ix
            contraction_ix += 1
            v[lower_ix][1] = contraction_ix
            v[lower_ix + 1][0] = contraction_ix
            contraction_ix += 1
        elif ix == len(mps1.tensors) - 1:
            # Assign contractions for the rightmost tensors.
            # Contract the right leg of the last 3-leg tensor with the horizontal legs
            # on the rightmost 2-leg tensor. Do that for upper and lower. Then contract
            # the vertical legs.
            v[upper_ix - 1][-1] = contraction_ix
            v[upper_ix][0] = contraction_ix
            contraction_ix += 1
            v[lower_ix - 1][-1] = contraction_ix
            v[lower_ix][0] = contraction_ix
            contraction_ix += 1
            v[upper_ix][1] = contraction_ix
            v[lower_ix][1] = contraction_ix
        else:
            # Assign contractions for a middle tensor.
            # The left contractions with the previous tensor are alread assigned, starting 
            # with the case where ix == 0. So, contract the vertical leg, and then the
            # horizontal legs contract with the neighbor.
            v[upper_ix][1] = contraction_ix
            v[lower_ix][1] = contraction_ix
            contraction_ix += 1
            v[upper_ix][-1] = contraction_ix
            v[upper_ix + 1][0] = contraction_ix
            contraction_ix += 1
            v[lower_ix][-1] = contraction_ix
            v[lower_ix + 1][0] = contraction_ix
            contraction_ix += 1
    inner_product = ncon(tensor_list, v)
    return inner_product
