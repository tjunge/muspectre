/**
 * @file   cell_split.cc
 *
 * @author Ali FalsafiTill Junge <ali.faslafi@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  Implementation for cell base class
 *
 * Copyright © 2017 Till Junge
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

#include "cell/cell_base.hh"
#include "cell/cell_split.hh"
#include "common/ccoord_operations.hh"
#include "common/iterators.hh"
#include "common/tensor_algebra.hh"

#include <sstream>
#include <algorithm>


namespace muSpectre {

    /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  CellSplit<DimS, DimM>::CellSplit(Projection_ptr projection)
    :Parent(projection), is_cell_splitted{SplittedCell::yes}{}

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::check_material_coverage(){
    auto nb_pixels = CcoordOps::get_size(this->subdomain_resolutions);
    std::vector<Real> pixel_assigned_ratio(nb_pixels, 0.0);
    for (auto & mat: this->materials) {
      for (auto & pixel: *mat) {
        auto index = CcoordOps::get_index(this->subdomain_resolutions,
                                          this->subdomain_locations,
                                          pixel);
        pixel_assigned_ratio.at(index) += mat->get_assigned_ratio(pixel);
      }
    }
    std::vector<Ccoord> over_assigned_pixels;
    std::vector<Ccoord> under_assigned_pixels;
    for (size_t i = 0; i < nb_pixels; ++i) {
      if (pixel_assigned_ratio.at(i) > 1.0) {
        over_assigned_pixels.push_back(CcoordOps::get_ccoord(this->subdomain_resolutions,
                                                             this->subdomain_locations, i));
      }else if (pixel_assigned_ratio.at(i) < 1.0) {
        under_assigned_pixels.push_back(CcoordOps::get_ccoord(this->subdomain_resolutions,
                                                              this->subdomain_locations, i));
      }
    }
     if (over_assigned_pixels.size() != 0) {
       std::stringstream err {};
       err << "Execesive material is assigned to the following pixels: ";
       for (auto & pixel: over_assigned_pixels) {
         err << pixel << ", ";
       }
       err << "and that cannot be handled";
       throw std::runtime_error(err.str());
     }
     if (under_assigned_pixels.size() != 0) {
       std::stringstream err {};
       err << "Insufficient material is assigned to the following pixels: ";
       for (auto & pixel: under_assigned_pixels) {
         err << pixel << ", ";
       }
       err << "and that cannot be handled";
       throw std::runtime_error(err.str());
     }
  }
    /* ---------------------------------------------------------------------- */
  //this piece of code handles the evaluation of stress an dtangent matrix
  //in case the cells have materials in which pixels are partially composed of
  //diffferent materials.

  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::evaluate_stress_tangent(StrainField_t & F){
    evaluate_split_stress_tangent(F)
  }

  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::evaluate_split_stress_tangent(StrainField_t & F){
    if (this->initialised == false) {
      this->initialise();
    }
    //! High level compatibility checks
    if (grad.size() != this->F.size()) {
      throw std::runtime_error("Size mismatch");
    }
    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    // Here we should first set P and K matrixes equal to zeros first
    this->set_p_k_zero()

    for (auto & mat: this->materials) {
      mat->compute_stresses_tangent(grad, this->P, this->K.value(),
                                    this->form, this->is_cell_splitted);
    }
    }

  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::set_p_k_zero(StrainField_t & F){
    auto nb_pixels = CcoordOps::get_size(this->subdomain_resolutions);
    
  }

} //muspectre
