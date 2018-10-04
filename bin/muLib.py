#!/usr/bin/env python3
"""
file   muLib.py

@author Till Junge <till.junge@epfl.ch>

@date   04 Oct 2018

@brief  Runner for the µLib server backend

@section LICENSE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""


import sys
import os
import argparse


from python_test_imports import µ


def parse_args():
    parser = argparse.ArgumentParser(description="Process a mulib calculation")

    parser.add_argument("path", metavar='PATH', type=str,
                        help = "path to netcdf-3 input file")
    parser.add_argument("newton_tol", metavar="NEWTON_TOL", type=float,
                        help="tolerance for newton loop")
    parser.add_argument("equil_tol", metavar="EQUIL_TOL", type=float,
                        help="tolerance for stress convergence")
    parser.add_argument("cg_tol", metavar="CG_TOL", type=float,
                        help="tolerance for congugate gradient solver")
    parser.add_argument("max_iter", metavar="MAXITER", type=int,
                        help="Max number of iterations")
    parser.add_argument("-v", "--verbose", action='store_true')

    return parser.parse_args()

def run(args):
    return µ.solvers.muLib(args.path, args.newton_tol, args.equil_tol,
                          args.cg_tol, args.max_iter, int(args.verbose))

def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
