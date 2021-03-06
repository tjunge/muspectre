#!/usr/bin/env python3
"""
file   small_case.py

@author Till Junge <till.junge@epfl.ch>

@date   12 Jan 2018

@brief  small case for debugging

@section LICENSE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "language_bindings/python"))
import muSpectre as µ


resolution = [51, 51]
center = np.array([r//2 for r in resolution])
incl = resolution[0]//5

lengths = [7., 5.]
formulation = µ.Formulation.small_strain

rve = µ.Cell(resolution, lengths, formulation)
hard = µ.material.MaterialLinearElastic1_2d.make(
    rve, "hard", 10e9, .33)
soft = µ.material.MaterialLinearElastic1_2d.make(
    rve, "soft",  70e9, .33)


for i, pixel in enumerate(rve):
    if np.linalg.norm(center - np.array(pixel),2)<incl:
    #if (abs(center - np.array(pixel)).max()<incl or
    #    np.linalg.norm(center/2 - np.array(pixel))<incl):
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)

tol = 1e-5
cg_tol = 1e-8

Del0 = np.array([[.0, .0],
                 [0,  .03]])
if formulation == µ.Formulation.small_strain:
    Del0 = .5*(Del0 + Del0.T)
maxiter = 401
verbose = 2

for solvclass in (µ.solvers.SolverCG,
                  µ.solvers.SolverCGEigen,
                  µ.solvers.SolverBiCGSTABEigen,
                  µ.solvers.SolverGMRESEigen,
                  µ.solvers.SolverDGMRESEigen,
                  µ.solvers.SolverMINRESEigen):
    print()
    try:
        solver = solvclass(rve, cg_tol, maxiter, verbose=False)
        r = µ.solvers.newton_cg(rve, Del0, solver, tol, verbose)
        print("nb of {} iterations: {}".format(solver.name(), r.nb_fev))
    except RuntimeError as err:
        print(err)
    try:
        solver = solvclass(rve, cg_tol, maxiter, verbose=False)
        r = µ.solvers.de_geus(rve, Del0, solver, tol, verbose)
        print("nb of {} iterations: {}".format(solver.name(), r.nb_fev))
    except RuntimeError as err:
        print(err)


# print(r.grad.T[:3])
# print(r.stress.T[:3])
# 
# print(r.grad.T.shape)
# import matplotlib.pyplot as plt
# stress = r.stress.T.reshape(*resolution, 2, 2)
# def comp_von_mises(arr):
#     out_arr = np.zeros(resolution)
#     s11 = arr[:,:,0,0]
#     s22 = arr[:,:,1,1]
#     s21_2 = arr[:,:,0,1]*arr[:,:,1,0]
# 
#     out_arr[:] = np.sqrt(.5*((s11-s22)**2) + s11**2 + s22**2 + 6*s21_2)
#     return out_arr
# 
# von_mises = comp_von_mises(stress)
# plt.pcolormesh(von_mises)
# plt.colorbar()
# plt.show()
