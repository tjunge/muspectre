"""
file   python_field_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   06 Jul 2018

@brief  tests the python bindings for fieldcollections, fields, and statefields

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

class FieldCollection_Check(unittest.TestCase):
    """Because field collections do not have python-accessible
    constructors, this test creates a problem with a material with
    statefields

    """
    def setUp(self):
        self.resolution = [3, 3]
        self.lengths = [1.58, 5.87]
        self.formulation = µ.Formulation.finite_strain
        self.sys = µ.Cell(self.resolution,
                          self.lengths,
                          self.formulation)
        self.dim = len(self.lengths)
        #self.mat = µ.material.MaterialHyperElastoPlastic
