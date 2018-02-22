"""iTEBD code to find the ground state of
the Heisenberg model on an infinite 2D honeycomb lattice.

J. D. Reger, J. A. Riera, and A. P. Young, J. Phys. Condens. Matter 1, 1855 (1989)
Energy E = -0.54455
Staggered Magnetization: m = 0.22

Frank Pollmann, frank.pollmann@tum.de"""

import numpy as np
import tps
import mps

def run(chi_mps, chi_tps, h, n_imaginary, n_boundary):
    """Use the simplified update to get the TPS"""
    print("Parameters: ", locals())
    # initialize Neel state
    G, s = tps.init_tps_product_state(2, 3, [1, 0])
    print("Simplified update with chi_tps = {c:d}".format(c=chi_tps))

    for delta in [0.1, 0.01, 0.001]:
        print("--> delta =", delta)
        N = int(n_imaginary / np.sqrt(delta))
        U_bond, H_bond = tps.init_heisenberg(h, 3, delta)
        for i in range(N):
            tps.sweep_tps(G, s, U_bond, chi_tps)

    ############ Attach the Lambda matrices symmetrically
    T_list = []
    for i_site in range(2):
        T = G[i_site]
        for b in range(3):
            T = tps.scale_axis(T, np.sqrt(s[b]), axis=1 + b)
        T_list.append(T)

    print("Getting energy and magnetization")

    ########################### Initiate the environment
    chi = T.shape[1]
    B_top, s_top = mps.init_mps_product_state(chi**2, [0, 0])
    B_bot, s_bot = mps.init_mps_product_state(chi**2, [0, 0])

    #################### Operator to measure m_staggered
    s0 = np.array([[1., 0.], [0., 1.]])
    sz = np.array([[1., 0.], [0., -1.]]) / 2.
    M = [1./3. * np.reshape(np.kron(s0, sz) - np.kron(sz, s0), (2, 2, 2, 2)) for i in range(3)]

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

        E = 3. / 2. * np.mean(expval_list[0])
        m = 3. / 2. * np.mean(np.abs(expval_list[1]))
        print("--> chi_boundary = {chi:2.0f}, E = {E:2.5f} (dE= {dE:2.2e}), m = {m:2.5f} (dm= {dm:2.2e})".
            format(chi=chi_boundary, E=E, m=m, dE=E + 0.54455, dm=m - 0.22))


if __name__ == "__main__":
    # get parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--chi_tps", type=int, default=3, help="max bond dimension of TPS")
    parser.add_argument("-c", "--chi_mps", type=int, default=20, help="max bond dimension of boundary MPS")
    parser.add_argument("-s", "--staggered_field", type=float, default=0., help="staggered magnetic field")
    parser.add_argument("-N", "--nupdates_imag", type=int, default=100, help="number of imaginary time steps")
    parser.add_argument("-n", "--nupdates_boundary", type=int, default=40, help="number of steps 'evolving' the boundary MPS")
    args = parser.parse_args()

    run(args.chi_mps, args.chi_tps, args.staggered_field, args.nupdates_imag, args.nupdates_boundary)
