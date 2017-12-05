/**
 * file   fftw_engine.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  FFT engine using FFTW
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <fftw3.h>

#include "fft/fft_engine_base.hh"

#ifndef FFTW_ENGINE_H
#define FFTW_ENGINE_H

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class FFTW_Engine: public FFT_Engine_base<DimS, DimM>
  {
  public:
    using Parent = FFT_Engine_base<DimS, DimM>;
    using Ccoord = typename Parent::Ccoord;
    using Workspace_t = typename Parent::Workspace_t;
    using Field_t = typename Parent::Field_t;
    //! Default constructor
    FFTW_Engine() = delete;

    //! Constructor with system sizes
    FFTW_Engine(Ccoord sizes);

    //! Copy constructor
    FFTW_Engine(const FFTW_Engine &other) = delete;

    //! Move constructor
    FFTW_Engine(FFTW_Engine &&other) = default;

    //! Destructor
    virtual ~FFTW_Engine() noexcept;

    //! Copy assignment operator
    FFTW_Engine& operator=(const FFTW_Engine &other) = delete;

    //! Move assignment operator
    FFTW_Engine& operator=(FFTW_Engine &&other) = default;

    // compute the plan, etc
    void initialise(FFT_PlanFlags plan_flags);

    //! forward transform
    Workspace_t & fft(const Field_t & field);

    //! inverse transform
    void ifft(Field_t & field) const;

  protected:
    Ccoord hermitian_sizes;
    fftw_plan plan_fft{};
    fftw_plan plan_ifft{};
  private:
  };

}  // muSpectre

#endif /* FFTW_ENGINE_H */
