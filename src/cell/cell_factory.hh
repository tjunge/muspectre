/**
 * @file   cell_factory.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Dec 2017
 *
 * @brief  Cell factories to help create cells with ease
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef CELL_FACTORY_H
#define CELL_FACTORY_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_base.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/projection_small_strain.hh"
#include "fft/fftw_engine.hh"

#ifdef WITH_MPI
#include "common/communicator.hh"
#include "fft/fftwmpi_engine.hh"
#endif

#include <memory>

namespace muSpectre {


  /**
   * Create a unique ptr to a Projection operator (with appropriate
   * FFT_engine) to be used in a cell constructor
   */
  template <Dim_t DimS, Dim_t DimM,
            typename FFTEngine=FFTWEngine<DimS>>
  inline
  std::unique_ptr<ProjectionBase<DimS, DimM>>
  cell_input(Ccoord_t<DimS> resolutions,
               Rcoord_t<DimS> lengths,
               Formulation form) {
    auto fft_ptr{
      std::make_unique<FFTEngine>(resolutions,
                                  dof_for_formulation(form, DimS))};
    switch (form)
      {
      case Formulation::finite_strain: {
        using Projection = ProjectionFiniteStrainFast<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr), lengths);
        break;
      }
      case Formulation::small_strain: {
        using Projection = ProjectionSmallStrain<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr), lengths);
        break;
      }
      default: {
        throw std::runtime_error("unknow formulation");
        break;
      }
    }
  }


  /**
   * convenience function to create a cell (avoids having to build
   * and move the chain of unique_ptrs
   */
  template <size_t DimS, size_t DimM=DimS,
            typename Cell=CellBase<DimS, DimM>,
            typename FFTEngine=FFTWEngine<DimS>>
  inline
  Cell make_cell(Ccoord_t<DimS> resolutions,
                 Rcoord_t<DimS> lengths,
                 Formulation form) {

    auto && input = cell_input<DimS, DimM, FFTEngine>(resolutions, lengths,
                                                      form);
    auto cell{Cell{std::move(input)}};
    return cell;
  }

#ifdef WITH_MPI

  /**
   * Create a unique ptr to a parallel Projection operator (with appropriate
   * FFT_engine) to be used in a cell constructor
   */
  template <Dim_t DimS, Dim_t DimM,
  typename FFTEngine=FFTWMPIEngine<DimS>>
  inline
  std::unique_ptr<ProjectionBase<DimS, DimM>>
  parallel_cell_input(Ccoord_t<DimS> resolutions,
                      Rcoord_t<DimS> lengths,
                      Formulation form,
                      const Communicator & comm) {
    auto fft_ptr{std::make_unique<FFTEngine>(resolutions,
                                             dof_for_formulation(form, DimM),
                                             comm)};
    switch (form)
    {
      case Formulation::finite_strain: {
        using Projection = ProjectionFiniteStrainFast<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr), lengths);
        break;
      }
      case Formulation::small_strain: {
        using Projection = ProjectionSmallStrain<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr), lengths);
        break;
      }
      default: {
        throw std::runtime_error("unknown formulation");
        break;
      }
    }
  }


  /**
   * convenience function to create a parallel cell (avoids having to build
   * and move the chain of unique_ptrs
   */
  template <size_t DimS, size_t DimM=DimS,
  typename Cell=CellBase<DimS, DimM>,
  typename FFTEngine=FFTWMPIEngine<DimS>>
  inline
  Cell make_parallel_cell(Ccoord_t<DimS> resolutions,
                          Rcoord_t<DimS> lengths,
                          Formulation form,
                          const Communicator & comm) {

    auto && input = parallel_cell_input<DimS, DimM, FFTEngine>(resolutions,
                                                               lengths,
                                                               form, comm);
    auto cell{Cell{std::move(input)}};
    return cell;
  }

#endif /* WITH_MPI */

}  // muSpectre

#endif /* CELL_FACTORY_H */
