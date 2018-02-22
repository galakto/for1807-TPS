"""Infinite Tensor-product state routines

Conventions:
z is the coordination number, e.g. z=3 for the Honeycomb lattice.

For the Tensor product state:
G[i] is the bulk tensor at site i of the unit cell,
    with indices [i, a, b, c, ...] for 1 physical leg and z bond legs,
    in "Gamma" form without any singular values multiplied to it.
s[z] are the singular values around the z bonds of G[0]

H_bond[z] is the bond hamiltonian at bond z around G[0].

For the boundary MPS: B and s as in the file mps.py
"""

import numpy as np
from scipy.linalg import expm
from scipy.linalg import svd
from mps import sweep as sweep_mps


def init_tps_product_state(d, z, state):
    G = []
    for st_i in state:
        G_i = np.zeros([d] + [1] * z)
        G_i[st_i, 0, ...] = 1.
        G.append(G_i)
    s = []
    for i in range(z):
        s.append(np.ones(1))
    return G, s


def init_ising(J, hx, z, delta):
    """ Returns bond hamiltonian and bond time-evolution"""
    s0 = np.array([[1., 0.], [0., 1.]])
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    d = 2
    U_bond = []
    H_bond = []
    for i in range(z):
        H = -J * np.kron(sz, sz) - hx * (np.kron(sx, s0) + np.kron(s0, sx)) / z
        H_bond.append(np.reshape(H, (d, d, d, d)))
        U_bond.append(np.reshape(expm(-delta * H), (d, d, d, d)))
    return U_bond, H_bond


def init_heisenberg(h, z, delta):
    """ Returns bond hamiltonian and bond time-evolution"""
    s0 = np.array([[1., 0.], [0., 1.]])
    sx = np.array([[0., 1.], [1., 0.]]) / 2.
    sy = np.array([[0., -1j], [1j, 0.]]) / 2.
    sz = np.array([[1., 0.], [0., -1.]]) / 2.
    d = 2
    U_bond = []
    H_bond = []
    for i in range(z):
        H = np.kron(sx, sx) + np.real(np.kron(sy, sy)) + np.kron(
            sz, sz) + h * (np.kron(sz, s0) - np.kron(s0, sz)) / z
        H_bond.append(np.reshape(H, (d, d, d, d)))
        U_bond.append(np.reshape(expm(-delta * H), (d, d, d, d)))
    return U_bond, H_bond


def sweep_tps(G, s, U_bond, chi):
    """ Perform the imaginary time evolution of an TPS with coordination number z using the simplified update."""
    GA, GB = G
    d = GA.shape[0]
    z = len(s)

    # Perform the imaginary time evolution on z bonds
    for bond in range(0, z):
        # Decorate Tensors with lambdas
        for b in range(0, z):
            GA = scale_axis(GA, s[b], axis=b + 1)
        for b in range(1, z):
            GB = scale_axis(GB, s[b], axis=b + 1)

        # Construct theta
        theta = np.tensordot(GA, GB, axes=(1, 1))  # i b c d ... j b' c' d' ...
        theta = np.tensordot(
            U_bond[bond], theta, axes=([2, 3], [0, z]))  # i' j' b c d ... b' c' d' ...
        theta = theta / np.linalg.norm(theta)
        dimbonds = list(GA.shape[1:])
        tr = [0] + list(range(2, z + 1)) + [1] + list(range(z + 1, 2 * z))
        theta = np.transpose(theta, tr)  # i b c d ... j b' c' d' ...
        theta = np.reshape(theta, (d * np.prod(dimbonds[1:]), d * np.prod(dimbonds[1:])))
        # Singular value decomposition
        X, Y, Z = svd(theta, full_matrices=0, lapack_driver='gesvd')
        Z = Z.T  # transpose
        # truncate
        chi_new = min(np.sum(Y > 10.**(-10)), chi)
        X = X[:, :chi_new]  # (i' b c d ... ) a
        Y = Y[:chi_new] / np.linalg.norm(Y[:chi_new])  # new singular values on bond a
        Z = Z[:, :chi_new]  # (j' b' c' d' ... ) a

        # Get the new tensors and "rotate" the bonds
        GA = np.reshape(X, [d] + dimbonds[1:] + [chi_new])  # i' b c d ... a
        GB = np.reshape(Z, [d] + dimbonds[1:] + [chi_new])  # j' b' c' d' ... a
        for b in range(1, z):
            GA = scale_axis(GA, s[b]**(-1), axis=b)
            GB = scale_axis(GB, s[b]**(-1), axis=b)
        # rotate the list `s` in same fashion as bulk
        del s[0]  # delete old singular values
        s.append(Y[:chi_new] / np.linalg.norm(Y[0:chi_new]))  # new singular values at end
        # (continue with next bond, rotate z times -> back to original)
    G[0] = GA
    G[1] = GB


def apply_T_TEBD(B, s, chi, Z_0, n_power=50):
    """ Use the TEBD sweep to apply rows of TPS to a boundary MPS.
        A number of blank sweeps are performed to obtain a canonical form.
    """
    d = np.shape(B[0])[0]
    for i in range(n_power):
        sweep_mps(B, s, 2 * [Z_0], chi, [0, 1])
    for i in range(10):
        identity = np.reshape(np.eye(d**2), (d, d, d, d))
        sweep_mps(B, s, 2 * [identity], chi, [0, 1])


def exp_value_infinite(B_top, B_bot, Z_0, Z_I, n_power=50):
    """ Get expectation value from two boundary MPS """
    C_t = np.tensordot(B_top[0], B_top[1], axes=(2, 1))  # i a j c
    C_b = np.tensordot(B_bot[0], B_bot[1], axes=(2, 1))  # i' a' j' c'

    T_0 = np.tensordot(C_t, Z_0, axes=([0, 2], [0, 1]))  # a c i' j'
    T_0 = np.tensordot(T_0, C_b, axes=([2, 3], [0, 2]))  # a c a' c'
    T_0 = np.transpose(T_0, (0, 2, 1, 3))  # a a' c c'

    T_I = np.tensordot(C_t, Z_I, axes=([0, 2], [0, 1]))  # a c i' j'
    T_I = np.tensordot(T_I, C_b, axes=([2, 3], [0, 2]))  # a c a' c'
    T_I = np.transpose(T_I, (0, 2, 1, 3))  # a a' c c'

    v_l = np.ones((C_t.shape[1], C_b.shape[1])) # a a'
    for j in range(n_power):
        v_l = np.tensordot(v_l, T_0, axes=([0, 1], [0, 1]))  # a a '
        v_l = v_l / np.linalg.norm(v_l)

    v_r = np.ones((C_t.shape[3], C_b.shape[3])) # c c'
    for j in range(n_power):
        v_r = np.tensordot(T_0, v_r, axes=([2, 3], [0, 1]))  # c c'
        v_r = v_r / np.linalg.norm(v_r)

    N = np.tensordot(v_l, T_0, axes=([0, 1], [0, 1]))
    N = np.tensordot(N, v_r, axes=([0, 1], [0, 1]))

    E = np.tensordot(v_l, T_I, axes=([0, 1], [0, 1]))
    E = np.tensordot(E, v_r, axes=([0, 1], [0, 1]))
    return E / N


def scale_axis(G, s, axis):
    """Apply a "diagonal matrix" s to a certain axis/leg of a tensor
    e.g., result[a,b,c,...] = G[a,b,c,...] s[b] for axis = 1 (no sum!)
    """
    result = np.tensordot(G, np.diag(s), axes=(axis, 0))  # a, c, d, ..., b'
    # transpose back to same order of legs as before
    tr = list(range(G.ndim))
    tr[axis:axis] = [G.ndim - 1]
    return np.transpose(result, tr[:-1])
