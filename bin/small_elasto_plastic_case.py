#!/usr/bin/env python3
"""
file   small_case.py

@author Till Junge <till.junge@epfl.ch>

@date   12 Jan 2018

@brief  small case for debugging elasto-plasticity

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
formulation = µ.Formulation.finite_strain

rve = µ.Cell(resolution, lengths, formulation)
hard = µ.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "hard", 10e9, .33, 1e6, h=.01)
soft = µ.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "soft",  70e9, .33, 1e6, h=.01)


for i, pixel in enumerate(rve):
    if np.linalg.norm(center - np.array(pixel),2)<incl:
    #if (abs(center - np.array(pixel)).max()<incl or
    #    np.linalg.norm(center/2 - np.array(pixel))<incl):
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)

tol = 1e-4
cg_tol = 1e-3

Del0 = np.array([[.0, .0],
                 [0,  1e-2]])
if formulation == µ.Formulation.small_strain:
    Del0 = .5*(Del0 + Del0.T)
maxiter = 401
verbose = 2

solver = µ.solvers.SolverCG(rve, cg_tol, maxiter, verbose=True)
r = µ.solvers.newton_cg(rve, Del0, solver, tol, verbose)
print("nb of {} iterations: {}".format(solver.name(), r.nb_fev))
