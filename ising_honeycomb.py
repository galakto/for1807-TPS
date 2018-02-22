"""iTEBD code to find the ground state of
the Ising model on an infinite 2D honeycomb lattice.

Frank Pollmann, frank.pollmann@tum.de

Compare to QMC results: Henk W. J. Bloete and Youjin Deng Phys. Rev. E 66, 066110
Critical field at gc = 2.13250(4)
"""

import numpy as np
import tps
import mps


def run(chi_mps, chi_tps, J, hx, n_imaginary, n_boundary):
    """Use the simplified update to get the TPS"""
    print("Parameters: ", locals())
    # initialize FM state
    G, s = tps.init_tps_product_state(2, 3, [0, 0])

    for delta in [0.1, 0.01, 0.001]:
        print("--> delta =", delta)
        N = int(n_imaginary / np.sqrt(delta))
        U_bond, H_bond = tps.init_ising(J, hx, 3, delta)
        for i in range(N):
            tps.sweep_tps(G, s, U_bond, chi_tps)

    ############ Attach the Lambda matrices symmetrically
    T_list = []
    for i_site in range(2):
        T = G[i_site]
        for b in range(3):
            T = tps.scale_axis(T, np.sqrt(s[b]), axis=1 + b)
        T_list.append(T)

    print("\nGetting energy and magnetization")

    ########################### Initiate the environment
    chi = T.shape[1]
    B_top, s_top = mps.init_mps_product_state(chi**2, [0, 0])
    B_bot, s_bot = mps.init_mps_product_state(chi**2, [0, 0])

    #################### Operator to measure <psi|Sz|psi>
    s0 = np.array([[1., 0.], [0., 1.]])
    sz = np.array([[1., 0.], [0., -1.]])
    M = [1. / 3. * np.reshape(np.kron(s0, sz) + np.kron(sz, s0), (2, 2, 2, 2)) for i in range(3)]

    ################################# Expectation values
    d = G[0].shape[0]
    for chi_boundary in list(range(5, chi_mps, 5)) + list(2 * [chi_mps]):
        expval_list = [[], []]

        for bond in [0, 1, 2]:
            # index notation assumes `bond` is `a`
            Z = np.tensordot(T_list[0], T_list[1], axes=(1, 1))  # i b c j b' c'
            Z0 = np.tensordot(Z, np.conj(Z), axes=([0, 3], [0, 3]))  # b c b' c' b* c* b'* c'*
            Z0 = np.transpose(Z0, (0, 4, 1, 5, 2, 6, 3, 7))  # b b* c c* b' b'* c' c'*
            Z0 = np.reshape(Z0, (chi**2, chi**2, chi**2, chi**2))  # (b b*) (c c*) (b' b'*) (c' c'*)
            Z0 = np.transpose(Z0, (1, 2, 0, 3))  # (c c*) (b' b'*) (b b*) (c' c'*)

            tps.apply_T_TEBD(B_top, s_top, chi_boundary, np.transpose(Z0, (2, 3, 0, 1)), n_power=n_boundary)
            tps.apply_T_TEBD(B_bot, s_bot, chi_boundary, Z0, n_power=n_boundary)

            for k, O in enumerate([H_bond, M]):
                Z1 = np.tensordot(O[bond], Z, axes=([2, 3], [0, 3]))  # i j b c b' c'
                Z1 = np.tensordot(Z1, np.conj(Z), axes=([0, 1], [0, 3]))  # b c b' c' b* c* b'* c'*
                Z1 = np.transpose(Z1, (0, 4, 1, 5, 2, 6, 3, 7))  # b b* c c* b' b'* c' c'*
                Z1 = np.reshape(Z1, (chi**2, chi**2, chi**2, chi**2))  # (b b*) (c c*) (b' b'*) (c' c'*)
                Z1 = np.transpose(Z1, (1, 2, 0, 3))  # (c c*) (b' b'*) (b b*) (c' c'*)
                expval_list[k].append(tps.exp_value_infinite(B_top, B_bot, Z0, Z1, n_power=n_boundary))

            # rotate T tensors to get to next bond
            T_list[0] = np.transpose(T_list[0], (0, 3, 1, 2))  # i c a b
            T_list[1] = np.transpose(T_list[1], (0, 3, 1, 2))  # j c' a' b'

        E = 3./2. * np.mean(expval_list[0])
        m = 3./2. * np.mean(np.abs(expval_list[1]))
        print("--> chi_boundary = {chi:2.0f}, E = {E:2.5f}, m = {m:2.5f}".format(
            chi=chi_boundary, E=E, m=m))


if __name__ == "__main__":
    # get parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--chi_tps", type=int, default=2, help="bond dimension of TPS")
    parser.add_argument("-c", "--chi_mps", type=int, default=20, help="max bond dimension of boundary MPS")
    parser.add_argument("-J", "--coupling", type=float, default=1., help="coupling strenth")
    parser.add_argument("-x", "--field_x", type=float, default=0.5, help="field in x direction")
    parser.add_argument("-N", "--nupdates_imag", type=int, default=100, help="number of imaginary time steps")
    parser.add_argument("-n", "--nupdates_boundary", type=int, default=10, help="number of steps 'evolving' the boundary MPS")
    args = parser.parse_args()

    run(args.chi_mps, args.chi_tps, args.coupling, args.field_x, args.nupdates_imag, args.nupdates_boundary)
