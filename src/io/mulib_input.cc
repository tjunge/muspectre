/**
 * file   mulib_input.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2018
 *
 * @brief  implementation of the µLib input file reader
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


#include "io/mulib_input.hh"
#include "materials/material_linear_elastic_generic.hh"
#include "cell/cell_base.hh"
#include "common/ccoord_operations.hh"

#include <ncException.h>

#include <sstream>
#include <system_error>
#include <numeric>
#include <vector>
#include <functional>

namespace muSpectre {

  MuLibInput::MuLibInput(filesystem::path path):
    path(path),
    file(path.native(), netCDF::NcFile::FileMode::read,
         netCDF::NcFile::FileFormat::classic)
  {
    file_error_code ec{};
    if (not filesystem::is_regular_file(path, ec)) {
      std::stringstream err_str{};
      err_str << "Path '" << path.native()
              << "' does not point to a regular file";
      throw filesystem::filesystem_error(err_str.str(), path, ec);
    }

    if (this->file.isNull()) {
      std::stringstream err_str{};
      err_str << "Could not open file '" << path.native() << "'";
      throw std::runtime_error(err_str.str());
    }

    // the microstructure is stored as an array of ints
    this->micro_structure = this->file.getVar("materialIndices");
    if (micro_structure.isNull()) {
      throw std::runtime_error("Could not read material indices");
    }

    const auto micro_structure_dims{micro_structure.getDims()};

    for (auto & dim: micro_structure_dims) {
      this->resolutions.push_back(dim.getSize());
      if (not (this->resolutions.back() % 2)) {
        --this->resolutions.back();
      }
    }

    // material labels contain a list of integer indices corresponding
    // to the material indices stored in the microstructure array.
    auto mat_labels_var{this->file.getVar("materialLabels")};

    if (mat_labels_var.isNull()) {
      throw std::runtime_error("Could not read material labels");
    }
    auto mat_labels_dims{mat_labels_var.getDims()};

    this->nb_materials = file.getDim("m").getSize();
    this->mat_labels.resize(this->nb_materials);
    mat_labels_var.getVar(this->mat_labels.data());

    // finally, the material parameters are given in form of a
    // stiffness tensor per material
    this->mat_params =this->file.getVar("materialStiffnesses");
    if (mat_params.isNull()) {
      throw std::runtime_error("Could not read stiffness tensors");
    }
    auto mat_params_dims{mat_params.getDims()};

    // make sure the size of the stiffness tensors make sense
    {
      const auto dim{this->get_dim()};
      if (not (size_t(vsize(dim)) == file.getDim("row").getSize()) or
          not (size_t(vsize(dim)) == file.getDim("col").getSize())) {
        std::stringstream err_str{};
        err_str << "expected stiffness tensors of shape " << vsize(dim)
                << " × " << vsize(dim) << " but got "
                << file.getDim("row").getSize() << " × "
                << file.getDim("col").getSize() << ".";
        throw std::runtime_error(err_str.str());
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  Dim_t MuLibInput::get_dim() const {
    return this->resolutions.size();
  }

  /* ---------------------------------------------------------------------- */
  const std::vector<Dim_t> & MuLibInput::get_resolutions() const {
    return this->resolutions;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void MuLibInput::setup_cell(CellBase<DimS, DimS>& cell) {
    using Material_t = MaterialLinearElasticGeneric<DimS, DimS>;
    using MatVec_t = std::vector<std::reference_wrapper<Material_t>>;
    MatVec_t materials{};
    std::map<Dim_t, size_t> lookup;

    constexpr auto t_size = size_t(vsize(DimS));

    for (int m{0}; m < this->get_nb_materials(); ++m) {
      const auto & label{this->mat_labels[m]};
      std::stringstream name_str{};
      name_str << "Material " << label << " in file '"
               << this->path.native() << "'";
      Eigen::Matrix<Real, t_size, t_size> tensor;

      // I can get away with neglecting the row-major netcdf format
      // because stiffness tensors are symmetric
      this->mat_params.getVar({size_t(m), 0, 0},
                              {1, t_size, t_size}, tensor.data());
      std::cout << "stiffness tensor for material " << label << std::endl;
      std::cout << tensor << std::endl << std::endl;
      materials.emplace_back( Material_t::make(cell, name_str.str(), tensor));
      lookup[label] = m;
    }


    // assign materials
    Ccoord_t<DimS> res{};
    for(auto && tup: akantu::zip(res, this->resolutions)) {
      std::get<0>(tup) = std::get<1>(tup);
    }
    std::vector<size_t> vec_pixel(DimS);
    for (auto && pixel: CcoordOps::Pixels<DimS>(res)) {
      for (int i{0}; i < DimS; ++i) {
        vec_pixel[i] = pixel[i];
      }
      Dim_t label;
      this->micro_structure.getVar(vec_pixel, &label);
      materials[lookup[label]].get().add_pixel(pixel);
    }


    cell.initialise(FFT_PlanFlags::measure);
  }

  template void MuLibInput::setup_cell(CellBase<  twoD,   twoD> &);
  template void MuLibInput::setup_cell(CellBase<threeD, threeD> &);

}  // muSpectre
