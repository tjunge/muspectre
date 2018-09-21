/**
 * file   mulib_input.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2018
 *
 * @brief  Class for handling the netcdf input file format for the minimal
 *         muLib configuration. In this configuration, a linear elastic problem
 *         is assumed, and the input file contains a per-pixel material label
 *         map of the two- or three-dimensional microstructure, a list of
 *         material properties in form of the elastic stiffness tensor as well
 *         as a map between labels and material properties. The input file
 *         format is netCDF3.
 *
 * Copyright © 2018 Till Junge
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

#ifndef MULIB_H
#define MULIB_H

#include "common/common.hh"
#include "common/utilities.hh"

#include <netcdf>
#include <Eigen/Dense>

#include <string>
#include <vector>

namespace muSpectre {

  /**
   * Forward declaration
   */
  template <Dim_t DimS, Dim_t DimM>
  class CellBase;


  /**
   * thin wrapper around a netcdf file for input from µLib
   */
  class MuLibInput
  {
  public:
    //! Default constructor
    MuLibInput() = delete;

    /**
     * Constructor from file path
     */
    MuLibInput(filesystem::path path);

    //! Copy constructor
    MuLibInput(const MuLibInput &other) = delete;

    //! Move constructor
    MuLibInput(MuLibInput &&other) = default;

    //! Destructor
    virtual ~MuLibInput() = default;

    //! Copy assignment operator
    MuLibInput& operator=(const MuLibInput &other) = delete;

    //! Move assignment operator
    MuLibInput& operator=(MuLibInput &&other) = default;

    //! returns the spatial dimension of the problem defined in the file
    Dim_t get_dim() const;

    //! returns the spatial dimension of the problem defined in the file
    const std::vector<Dim_t> & get_resolutions() const;

    //! returns the number of materials in the microstructure
    Dim_t get_nb_materials() const {return this->nb_materials;}
    /**
     * fills the materials for the provided Cell and initialises it
     */
    template <Dim_t DimS>
    void setup_cell(CellBase<DimS, DimS> & cell);


  protected:
    const filesystem::path path;
    netCDF::NcFile file;
    std::vector<Dim_t> resolutions{};
    Dim_t nb_materials{};
    netCDF::NcVar mat_params{};
    netCDF::NcVar micro_structure{};
    std::vector<Dim_t> mat_labels{};
  private:
  };


}  // muSpectre

#endif /* MULIB_H */
