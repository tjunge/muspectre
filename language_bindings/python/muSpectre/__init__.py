#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  Main entry point for muSpectre Python module

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Lesser Public License as
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


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import _muSpectre
from _muSpectre import (Formulation, get_domain_ccoord, get_domain_index,
                        get_hermitian_sizes, material, solvers)
import muSpectre.fft

_factories = {'fftw': ('CellFactory', False),
              'fftwmpi': ('FFTWMPICellFactory', True),
              'pfft': ('PFFTCellFactory', True),
              'p3dfft': ('P3DFFTCellFactory', True)}


def Cell(resolutions, lengths, formulation=Formulation.finite_strain,
         fft='fftw', communicator=None):
    """
    Instantiate a muSpectre Cell class.

    Parameters
    ----------
    resolutions: list
        Grid resolutions in the Cartesian directions.
    lengths: list
        Physical size of the cell in the Cartesian directions.
    formulation: Formulation
        Formulation for strains and stresses used by the solver. Options are
        `Formulation.finite_strain` and `Formulation.small_strain`. Finite
        strain formulation is the default.
    fft: string
        FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and 'p3dfft'.
        Default is 'fftw'.
    communicator: mpi4py communicator
        mpi4py communicator object passed to parallel FFT engines. Note that
        the default 'fftw' engine does not support parallel execution.


    Returns
    -------
    cell: object
        Return a muSpectre Cell object.
    """
    try:
        factory_name, is_parallel = _factories[fft]
    except KeyError:
        raise KeyError("Unknown FFT engine '{}'.".format(fft))
    try:
        factory = _muSpectre.__dict__[factory_name]
    except KeyError:
        raise KeyError("FFT engine '{}' has not been compiled into the "
                       "muSpectre library.".format(fft))
    if is_parallel:
        if MPI is None:
            raise RuntimeError('Parallel solver requested but mpi4py could'
                               ' not be imported.')
        if communicator is None:
            communicator = MPI.COMM_SELF
        return factory(resolutions, lengths, formulation,
                       MPI._handleof(communicator))
    else:
        if communicator is not None:
            raise ValueError("FFT engine '{}' does not support parallel "
                             "execution.".format(fft))
        return factory(resolutions, lengths, formulation)
