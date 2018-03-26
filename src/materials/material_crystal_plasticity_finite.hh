/**
 * @file   material_crystal_plasticity_finite.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Francesco Maresca <francesco.maresca@epfl.ch>
 *
 * @date   23 Feb 2018
 *
 * @brief finite strain crystal plasticity implementation
 *
 * Copyright © 2018 Till Junge, Francesco Maresca
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

#ifndef MATERIAL_CRYSTAL_PLASTICITY_FINITE_H
#define MATERIAL_CRYSTAL_PLASTICITY_FINITE_H

#include "materials/material_muSpectre_base.hh"
#include "common/field.hh"

#include <Eigen/Dense>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  class MaterialCrystalPlasticityFinite;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  struct MaterialMuSpectre_traits<MaterialCrystalPlasticityFinite<DimS, DimM, nb_slip>>
  {
    //! global field collection
    using GFieldCollection_t = typename
      MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! declare internal variables
    //! local field_collections used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;
    //! plastic deformation gradient Fₚ(t)
    using FpMap_t = StateFieldMap<MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, false>>;
    //! plastic slip rates dγᵅ/dt
    using GammadotMap_t = StateFieldMap<MatrixFieldMap<LFieldColl_t, Real, nb_slip, 1, false>>;
    //! critical resolved shear stresses (CRSS) τᵅy(t)
    using TauyMap_t = GammadotMap_t;
    //! plastic slips γᵅ(t)
    using GammaMap_t = MatrixFieldMap<LFieldColl_t, Real, nb_slip, 1, false>;
    //! euler angles
    using EulerMap_t = ArrayFieldMap<LFieldColl_t, Real, (DimM==3) ? 3 : 1, 1, true>;
    
    using InternalVariables = std::tuple<FpMap_t, GammadotMap_t, TauyMap_t, GammaMap_t, EulerMap_t>; 
  };

  /**
   * implements finite strain crystal plasticity
   */
  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  class MaterialCrystalPlasticityFinite:
    public MaterialMuSpectre<MaterialCrystalPlasticityFinite<DimS, DimM, nb_slip>, DimS, DimM>
  {
  public:

    //! base class
    using Parent = MaterialMuSpectre<MaterialCrystalPlasticityFinite, DimS, DimM, nb_slip>;

    //! type for stiffness tensor construction
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialCrystalPlasticityFinite>;

    //! Type of container used for storing eigenstrain
    using InternalVariables = typename traits::InternalVariables;

    //! Hooke's law implementation
    using Hooke = typename
      MatTB::Hooke<DimM,
                   typename traits::StrainMap_t::reference,
                   typename traits::TangentMap_t::reference>;

    //! Type in which plastic deformation gradient is referenced 
    using Fp_ref = typename traits::FpMap_t::reference;
    //! Type in which slip rates are referenced
    using Gammadot_ref = typename traits::GammadotMap_t::reference;
    //! Type in which CRSS are referenced
    using Tauy_ref = typename traits::TauyMap_t::reference;
    //! Type in which accumulated slips are referenced
    using Gamma_ref = typename traits::GammaMap_t::reference;
    //! Type in which Euler angles are referenced
    using Euler_ref = typename traits::EulerMap_t::reference;
    //! Type in which slip directions and normals are given
    using SlipVecs = Eigen::Matrix<Real, nb_slip, DimM>;
    using SlipVecs_ref = Eigen::Ref<SlipVecs>;
    //! Type for second rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    //! Type for fourth rank tensors
    using T4_t = T4Mat<Real, DimM>;
    
    //! Default constructor
    MaterialCrystalPlasticityFinite() = delete;

    //! Construct by name, Bulk modulus, Shear modulus, Reference slip rate, Strain rate sensitivity, Initial CRSS, Initial hardening modulus, CRSS saturation value, Hardening exponent, Latent/Self-hardening ratio, Slip directions, Slip normals
    MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gammadot0, Real m_par, Real tauy0, Real h0, Real s_infty, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0);

    //! Copy constructor
    MaterialCrystalPlasticityFinite(const MaterialCrystalPlasticityFinite &other) = delete;

    //! Move constructor
    MaterialCrystalPlasticityFinite(MaterialCrystalPlasticityFinite &&other) = delete;

    //! Destructor
    virtual ~MaterialCrystalPlasticityFinite() = default;

    //! Copy assignment operator
    MaterialCrystalPlasticityFinite& operator=(const MaterialCrystalPlasticityFinite &other) = delete;

    //! Move assignment operator
    MaterialCrystalPlasticityFinite& operator=(MaterialCrystalPlasticityFinite &&other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Deformation Gradient
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref Gamma, Euler_ref Euler);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Deformation Gradient
     */
    template <class s_t>
    inline decltype(auto)
    evaluate_stress_tangent(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref Gamma, Euler_ref Euler);

    /**
     * return the internals tuple
     */
    InternalVariables & get_internals() {
      return this->internal_variables;};

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) override final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel,
                   const Eigen::Ref<Eigen::Array<Real, nb_euler, 1>> Euler);

  protected:
    constexpr static Int nb_euler{(DimM==3) ? 3 : 1};
    //! Storage for Fp
    using FpField_t =
      StateField<TensorField<typename traits::LFieldColl_t, Real, secondOrder, DimM>>;
    FpField_t & FpField;
    //! Storage for gammadot
    using GammadotField_t =
      StateField<MatrixField<typename traits::LFieldColl_t, Real, nb_slip, 1>>;
    GammadotField_t & GammadotField;
    //! Storage for tauy
    using TauyField_t = GamadotField_t;
    TauyField_t & TauyField;
    //! Storage for gamma
    using GammaField_t =
      MatrixField<typename traits::LFieldColl_t, Real, nb_slip, 1>;
    GammaField_t & GammaField;
    using EulerField_t =
      ArrayField<typename traits::LFieldColl_t, Real, nb_euler, 1>;
    EulerField_t & EulerField;
    //! bulk modulus
    Real bulk_m;
    Real shear_m;
    Real gammadot0;
    Real m_par;
    Real tauy0;
    Real h0;
    Real s_infty;
    Real a_par;
    Real q_n;
    //! Storage for slip directions
    alignas(16) Eigen::Matrix<Real, nb_slip, DimM> Slip0;
    //! Storage for slip plane normals
    alignas(16) Eigen::Matrix<Real, nb_slip, DimM> Normal0;
    T4_t C_el;
    
    //! tuple for iterable eigen_field
    InternalVariables internal_variables;
  private:
  };
  
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  template <class s_t>
  decltype(auto)
  MaterialCrystalPlasticityFinite<DimS, DimM, nb_slip>::
  evaluate_stress(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref Gamma, Euler_ref Euler) {
    Rotator<DimM> Rot(Euler);
    T2_t Floc{Rot.glob_to_loc(F)};
    std::array<T2_t, nb_slip> SchmidT;
    for (Int i{0}; i < nb_slip; ++i) {
      SchmidT[i]=this->Slip0.col(i)*this->Normal0.col(i).transpose();
    }

    // trial elastic deformation
    T2_t Fe_star{Floc*Fp.old().inverse()};
    T2_t CGe_star{Fe_star.transpose()*Fe_star};
    T2_t GLe_star{.5*(CGe_star - T2_t::Identity())};
    T2_t SPK_star{Matrices::tensmult(C_el,GLe_star)};

    std::array<Real, nb_slip> tau_star;
    // pl_corr is the plastic corrector
    std::array<T2_t, nb_slip> pl_corr;
    Eigen::Matrix<Real, nb_slip, nb_slip> pl_corr_proj;
    for (Int i{0}; i < nb_slip; ++i) {
      tau_star[i]=(CGe_star*SPK_star*SchmidT[i].transpose()).trace();
      pl_corr[i]=tensmult(C_el,.5*(CGe_star*SchmidT[i]+SchmidT[i].transpose()*CGe_star));
      for (Int j{0}; j < nb_slip; ++j) {
	pl_corr_proj(i,j)=(C_el*pl_corr[i]*SchmidT[j].transpose()).trace();
	}
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  template <class s_t, class eigen_s_t>
  decltype(auto)
  MaterialCrystalPlasticityFinite<DimS, DimM, nb_slip>::
  evaluate_stress_tangent(s_t && E, eigen_s_t && E_eig) {
    // using mat = Eigen::Matrix<Real, DimM, DimM>;
    // mat ecopy{E};
    // mat eig_copy{E_eig};
    // mat ediff{ecopy-eig_copy};
    // std::cout << "eidff - (E-E_eig)" << std::endl << ediff-(E-E_eig) << std::endl;
    // std::cout << "P1 <internal>" << std::endl << mat{std::get<0>(this->material.evaluate_stress_tangent(E-E_eig))} << "</internal>" << std::endl;
    // std::cout << "P2" << std::endl << mat{std::get<0>(this->material.evaluate_stress_tangent(std::move(ediff)))} << std::endl;
    return this->material.evaluate_stress_tangent(E-E_eig);
  }


}  // muSpectre

#endif /* MATERIAL_CRYSTAL_PLASTICITY_FINITE_H */
