""" iTEBD code to find the ground state of
the 1D Ising model on an infinite 1D chain.

Frank Pollmann, frank.pollmann@tum.de"""

import numpy as np
import mps

def run(chi, n_imaginary):
    """run imaginary time evolution for the Heisenberg chain with TEBD"""
    print("Parameters: ", locals())
    # Generate a Neel initial state
    B, s = mps.init_mps_product_state(2, [0, 1])

    for delta in [0.1, 0.01, 0.001]:
        N = int(n_imaginary / np.sqrt(delta))
        U_bond, H_bond = mps.init_heisenberg(2, delta)
        for i in range(N):
            mps.sweep(B, s, U_bond, chi)
        E = np.mean(mps.bond_expectation(B, s, H_bond))
        m = np.mean(np.abs(mps.site_expectation(B, s, 2 * [np.array([[1., 0.], [0., -1.]]) / 2])))
        print("--> delta = {D:.6f}, E = {E:2.6f}, (dE= {dE:2.2e}), m = {m:2.6f}".format(
            D=delta, E=E, dE=E + (np.log(2) - 0.25), m=m))


if __name__ == "__main__":
    # get parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chi", type=int, default=10, help="bond dimension of MPS")
    parser.add_argument( "-N", "--nupdates_imag", type=int, default=100, help="number of imaginary time steps")
    args = parser.parse_args()

    run(args.chi, args.nupdates_imag)
