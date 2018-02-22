"""Infinite MPS routines.

Conventions:
B[i] is tensor with axes [i,a,b] (for physical, left bond, right bond)
s[i] are singular values at bond between sites (i-1,i).
H_bond[i] is the bond hamiltonian between sites (i,i+1).
"""

import numpy as np
from scipy.linalg import svd
from scipy.linalg import expm


def init_mps_product_state(d, state):
    """ Returns Tensors and singular values of an iMPS representing a product state"""
    B = []
    s = []
    for st_i in state:
        B_i = np.zeros([d, 1, 1])
        B_i[st_i, 0, 0] = 1.
        B.append(B_i)
        s.append(np.ones(1))
    return B, s


def init_ising(J, hx, hz, L, delta):
    """ Returns bond hamiltonian and bond time-evolution"""
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    s0 = np.eye(2)
    d = 2
    U_bond = []
    H_bond = []
    for i in range(L):
        H = -J * np.kron(sz, sz) - hx * (np.kron(sx, s0) + np.kron(s0, sx)) / 2. - hz * (
            np.kron(sz, s0) + np.kron(s0, sz)) / 2.
        H_bond.append(np.reshape(H, (d, d, d, d)))
        U_bond.append(np.reshape(expm(-delta * H), (d, d, d, d)))
    return U_bond, H_bond


def init_heisenberg(L, delta):
    """ Returns bond hamiltonian and bond time-evolution"""
    sx = np.array([[0., 1.], [1., 0.]]) / 2.
    sy = np.array([[0., -1j], [1j, 0.]]) / 2.
    sz = np.array([[1., 0.], [0., -1.]]) / 2.
    d = 2
    U_bond = []
    H_bond = []
    for i in range(L):
        H = np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
        H_bond.append(np.reshape(H, (d, d, d, d)))
        U_bond.append(np.reshape(expm(-delta * H), (d, d, d, d)))
    return U_bond, H_bond


def bond_expectation(B, s, O_list):
    " Expectation value for a bond operator "
    E = []
    L = len(B)
    for i_bond in range(L):
        BB = np.tensordot(B[i_bond], B[np.mod(i_bond + 1, L)], axes=(2, 1))
        sBB = np.tensordot(np.diag(s[np.mod(i_bond, L)]), BB, axes=(1, 1))
        C = np.tensordot(sBB, O_list[i_bond], axes=([1, 2], [2, 3]))
        sBB = np.conj(sBB)
        E.append(np.squeeze(np.tensordot(sBB, C, axes=([0, 3, 1, 2], [0, 1, 2, 3]))).item())
    return E


def site_expectation(B, s, O_list):
    " Expectation value for a site operator "
    E = []
    L = len(B)
    for isite in range(0, L):
        sB = np.tensordot(np.diag(s[np.mod(isite, L)]), B[isite], axes=(1, 1))
        C = np.tensordot(sB, O_list[isite], axes=(1, 0))
        sB = sB.conj()
        E.append(np.squeeze(np.tensordot(sB, C, axes=([0, 1, 2], [0, 2, 1]))).item())
    return (E)


def entanglement_entropy(s):
    " Returns the half chain entanglement entropy "
    S = []
    for s_bond in s:
        x = s_bond[s_bond > 1.e-20]**2
        S.append(-np.inner(np.log(x), x))
    return S


def correlation_length(B):
    "Constructs the mixed transfermatrix and returns correlation length"
    from scipy.sparse.linalg import eigs as sparse_eigs
    chi = B[0].shape[1]
    L = len(B)

    T = np.tensordot(B[0], np.conj(B[0]), axes=(0, 0))  # a,b,a*,b*
    T = T.transpose(0, 2, 1, 3)  # a,a*,b,b*
    for i in range(1, L):
        T = np.tensordot(T, B[i], axes=(2, 1))  # a,a*,b*,i,b
        T = np.tensordot(T, np.conj(B[i]), axes=([2, 3], [1, 0]))  #a,a*,b,b*
    T = np.reshape(T, (chi**2, chi**2))

    # Obtain the 2nd largest eigenvalue
    eta = sparse_eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
    return -L / np.log(np.min(np.abs(eta)))


def sweep(B, s, U_bond, chi, even_odd=[0, 1]):
    """ Perform the imaginary time evolution of an MPS """
    L = len(B)
    d = B[0].shape[0]
    for k in even_odd:
        for i_bond in range(k, L, 2):
            ia = i_bond
            ib = np.mod(i_bond + 1, L)
            chia = B[ia].shape[1]
            chic = B[ib].shape[2]

            # Construct theta matrix and time evolution #
            theta = np.tensordot(B[ia], B[ib], axes=(2, 1))  # i a j c
            theta = np.tensordot(U_bond[i_bond], theta, axes=([2, 3], [0, 2]))  # i' j' a c
            theta = np.tensordot(np.diag(s[ia]), theta, axes=([1, 2]))  # a i' j' c
            theta = np.reshape(np.transpose(theta, (1, 0, 2, 3)),
                               (d * chia, d * chic))  # (i' a) (j' c)

            # Schmidt deomposition #
            X, Y, Z = svd(theta, full_matrices=0, lapack_driver='gesvd')
            # (Y is sorted descending)
            chi2 = min(np.sum(Y > 10.**(-10)), chi)

            Y = Y[:chi2]
            invsq = np.sqrt(sum(Y**2))
            X = X[:, :chi2]  # (i' a) b
            Z = Z[:chi2, :]  # b (j' c)

            # Obtain the new values for B and s #
            s[ib] = Y / invsq

            X = np.reshape(X, (d, chia, chi2))
            X = np.transpose(np.tensordot(np.diag(s[ia]**(-1)), X, axes=(1, 1)), (1, 0, 2))
            B[ia] = np.tensordot(X, np.diag(s[ib]), axes=(2, 0))
            B[ib] = np.transpose(np.reshape(Z, (chi2, d, chic)), (1, 0, 2))
