#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_crystal_plasticity_finite1.py

@author Till Junge <till.junge@epfl.ch>

@date   24 May 2018

@brief  tests for crystal plasticity material

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

import unittest
import numpy as np

from python_test_imports import µ

class MaterialCrystalPlasticityFinite1_Check(unittest.TestCase):
    """
    Check the bindings for MaterialCrystalPlasticityFinite
    """
    def setUp(self):
        self.bulk_m = 175e9;
        self.shear_m = 120e9;
        self.gamma_dot0 = 10e-2;
        self.m_par = .1;
        self.tau_y0 = 200e6;
        self.h0 = 0e9;
        self.delta_tau_y = 100e6;
        self.a_par = 0;
        self.q_n = 1.4;
        self.delta_t = 1e-3;
        self.dirs = np.array([[1., 0.]])
        self.norms = np.array([[0., 1.]])

        self.resolution = [1,1]
        self.lengths = [1, 1]
        self.formulation = µ.Formulation.finite_strain

        self.cell = µ.Cell(self.resolution,
                           self.lengths,
                           self.formulation)
        self.dim = len(self.lengths)

        self.mat = µ.material.MaterialCrystalPlasticityFinite_2d_1slip.make(
            self.cell,
            "crystal",
            self.bulk_m,
            self.shear_m,
            self.gamma_dot0,
            self.m_par,
            self.tau_y0,
            self.h0,
            self.delta_tau_y,
            self.a_par,
            self.q_n,
            self.dirs,
            self.norms,
            self.delta_t)

    def test_stressStrain(self):
        for pixel in self.cell:
            self.mat.add_pixel(pixel, np.array([[0.]]))

        self.cell.initialise()
        tau = list()
        gammas = np.linspace(5e-3, 5e-2, 10)
        for gamma in gammas:
            F = np.array([[1, gamma],
                          [0,     1]])

            stress = self.cell.evaluate_stress(F.T.reshape(-1))
            self.mat.save_history_variables()
            print ("stress:\n{}\n".format(stress))
            tau.append(stress[2])
        print("gamma = np.array([{}])".format(", ".join(
            ("{}".format(g) for g in gammas))))
        print("tau = np.array([{}])".format(", ".join(
            ("{}".format(t) for t in tau))))


if __name__ == '__main__':
    mat = MaterialCrystalPlasticityFinite1_Check()
    mat.setUp()
    mat.test_stressStrain()
