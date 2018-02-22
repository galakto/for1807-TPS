"""iTEBD code to find the ground state of
the 1D Ising model on an infinite 1D chain.

Frank Pollmann, frank.pollmann@tum.de"""

import numpy as np
import mps
from scipy import integrate


def run(chi, J, hx, hz, n_imaginary):
    """run imaginary time evolution for the ising chain with TEBD"""
    print("Parameters: ", locals())
    # Generate an FM initial state
    B, s = mps.init_mps_product_state(2, [0, 0])

    for delta in [0.1, 0.01, 0.001,0.0001]:
        N = int(n_imaginary / np.sqrt(delta))
        U_bond, H_bond = mps.init_ising(J, hx, hz, 2, delta)
        for i in range(N):
            mps.sweep(B, s, U_bond, chi)

        E = np.mean(mps.bond_expectation(B, s, H_bond))
        m = np.mean(mps.site_expectation(B, s, 2 * [np.diag([-1, 1])]))
        # Calculate exact groundstate energy
        if hz == 0 and np.abs(J) == 1:
            def f(k, hx):
                return -np.sqrt(1 + hx**2 - 2 * hx * np.cos(k)) / np.pi
            E0_exact = integrate.quad(f, 0, np.pi, args=(hx, ))[0]
            print("--> delta = {D:.6f}, E = {E:2.6f} (dE = {dE:2.2e}), m = {m:2.6f}".format(D=delta, E=E,dE=E - E0_exact,m=np.abs(m)))
        else:
            print("--> delta = {D:.6f}, m = {m:2.6f}, E = {E:2.6f}".format(D=delta, m=np.abs(m), E=E))

if __name__ == "__main__":
    # get parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chi", type=int, default=10, help="bond dimension of MPS")
    parser.add_argument("-J", "--coupling", type=float, default=1., help="coupling strength")
    parser.add_argument("-x", "--field_x", type=float, default=0.5, help="field in x direction")
    parser.add_argument("-z", "--field_z", type=float, default=0.0, help="field in z direction")
    parser.add_argument("-N", "--nupdates_imag", type=int, default=100, help="number of imaginary time steps")
    args = parser.parse_args()
    
    # run simulation
    run(args.chi, args.coupling, args.field_x, args.field_z, args.nupdates_imag)
