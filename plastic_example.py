#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append("build-clang-5.0/language_bindings/python")
import muSpectre as msp
import matplotlib.pyplot as plt

## currently, muSpectre is restricted to odd-numbered resolutions for
## reasons explained in T.W.J. de Geus, J. Vondřejc, J. Zeman,
## R.H.J. Peerlings, M.G.D. Geers, Finite strain FFT-based non-linear
## solvers made simple, Computer Methods in Applied Mechanics and
## Engineering, Volume 318, 2017
## https://doi.org/10.1016/j.cma.2016.12.032
# read phase indicator from micrograph: 0=soft, 1=hard
phase  = np.load('odd_image.npz')['phase']
phase = phase[:9, :9]

resolution = list(phase.shape)
center = np.array([r//2 for r in resolution])
incl = resolution[0]//5


## Domain dimensions
lengths = [float(r) for r in resolution]
## formulation (small_strain or finite_strain)
formulation = msp.Formulation.finite_strain

## build a computational domain
rve = msp.Cell(resolution, lengths, formulation)


def get_E_nu(bulk, shear):
    Young = 9*bulk*shear/(3*bulk + shear)
    Poisson = Young/(2*shear) - 1
    return Young, Poisson

E, nu = get_E_nu(.833, .386)
H = (2000.0e6/200.0e9,2.*2000.0e6/200.0e9)
tauy0 = ( 600.0e6/200.0e9,2.* 600.0e6/200.0e9)
## define the material properties of the matrix and inclusion
hard = msp.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "hard", E, nu, tauy0[1], H[1])
soft = msp.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "soft", E, nu, tauy0[0], H[0])

## assign each pixel to exactly one material
for i, pixel in enumerate(rve):
    if phase[pixel[0], pixel[1]] == 1:
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)

## define the convergence tolerance for the Newton-Raphson increment
tol = 1e-5
## tolerance for the solver of the linear cell
cg_tol = 1e-8
eq_tol = 1e-8


## Macroscopic strain
def get_Del0(eps_bars):
    def Del0(eps_bar):
        stretch = np.exp(np.sqrt(3.0)/2.0*eps_bar)
        Del0 = np.array([[stretch-1, 0.        ],
                         [     0.,  1/stretch-1]])
        #Del0 = np.array([[0, eps_bar],
        #                 [0, 0]])
        return Del0
    return [Del0(eps_bar) for eps_bar in eps_bars]

Del_gamma = .2 # total strain
nb_gamma = 250 # increments
del_gamma = Del_gamma/nb_gamma
Del0 = get_Del0((i*del_gamma for i in range(1, nb_gamma+1)))


maxiter = 500 ## for linear cell solver

## Choose a solver for the linear cells. Currently avaliable:
## SolverCG, SolverCGEigen, SolverBiCGSTABEigen, SolverGMRESEigen,
## SolverDGMRESEigen, SolverMINRESEigen.
## See Reference for explanations
solver = msp.solvers.SolverCG(rve, cg_tol, maxiter, verbose=False)


## Verbosity levels:
## 0: silent,
## 1: info about Newton-Raphson loop,
verbose = 2

## Choose a solution strategy. Currently available:
## de_geus: is described in de Geus et al. see Ref above
## newton_cg: classical Newton-Conjugate Gradient solver. Recommended
result = msp.solvers.de_geus(rve, Del0, solver, tol, eq_tol, verbose=verbose)

mean_stress = [res.stress.reshape(-1, 4).mean(axis=0) for res in result]

mean_tau = np.array([stress[2] for stress in mean_stress])
gammas = np.array([del0[0,1] for del0 in Del0])
plt.plot(gammas, mean_tau)
plt.show()

## visualise e.g., stress in y-direction
stress = result[-1].stress
## stress is stored in a flatten stress tensor per pixel, i.e., a
## dim^2 × prod(resolution_i) array, so it needs to be reshaped
stress = stress.T.reshape(*resolution, 2, 2)

plt.pcolormesh(stress[:, :, 1, 1])
plt.colorbar()
plt.show()

