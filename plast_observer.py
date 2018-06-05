import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./language_bindings/python/")
from numpy.linalg import norm
import muSpectre as µ



def hyper_curve(tau_y0 = 200e6,     h0 = 10e9, max_shear = 10e-2):
    bulk_m = 175e9;
    shear_m = 120e9;

    bulk_m = .833
    shear_m = .386
    Young = 9*bulk_m*shear_m/(3*bulk_m + shear_m)
    Poisson = Young/(2*shear_m) - 1
    print("E = {}, ν = {}".format(Young, Poisson))

    tolerance=1e-12
    resolution = [1,1]
    lengths = [1, 1]
    formulation = µ.Formulation.finite_strain

    cell = µ.Cell(resolution,
                  lengths,
                  formulation)
    dim = len(lengths)

    mat = µ.material.MaterialHyperElastoPlastic1_2d.make(
        cell, "hyper-elasto-plastic", 210e9, .3, tau_y0, h0)

    for pixel in cell:
        mat.add_pixel(pixel)

        cell.initialise()

        tau = list()
        sigma_xx=list()
        gammas=list()
        #tau_inc = list()
        #gamma_dot = list()
        shear_incr = .1e-4
        np_load = int(max_shear/shear_incr)
        F = np.eye(2)

        for step in range(np_load):
            F[0,1] += shear_incr
            gammas.append(F[0,1])

            stress = cell.evaluate_stress(F.T.reshape(-1))
            mat.save_history_variables()
            tau.append(stress[2])
            sigma_xx.append(stress[0])
    gammas = np.array(gammas).reshape(-1)
    tau = np.array(tau)

    return gammas, tau, sigma_xx

gammas, tau, sigma_xx = hyper_curve(tau_y0 = .003, h0 = .004, max_shear = 5e-3)
plt.plot(gammas, tau, label=r"$\tau$ default")
#plt.plot(gammas, sigma_xx, label=r"$\sigma_{{xx}}$ default")
pass
plt.legend(loc='best')
plt.xlabel('ε₁₂')
plt.ylabel('γ₁₂');

plt.show()
