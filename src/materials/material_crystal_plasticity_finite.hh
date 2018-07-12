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
#include "common/geometry.hh"
#include "common/statefield.hh"
#include "common/eigen_tools.hh"

#include <Eigen/Dense>

#include <cmath>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  class MaterialCrystalPlasticityFinite;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  struct MaterialMuSpectre_traits<MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>>
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
    using GammaDotMap_t = StateFieldMap<MatrixFieldMap<LFieldColl_t, Real, NbSlip, 1, false>>;
    //! critical resolved shear stresses (CRSS) τᵅy(t)
    using TauYMap_t = GammaDotMap_t;
    //! euler angles
    using EulerMap_t = ArrayFieldMap<LFieldColl_t, Real, (DimM==3) ? 3 : 1, 1, true>;

    //! dummies for debugging
    using DummyGammaDot_t = MatrixFieldMap<LFieldColl_t, Real, NbSlip, 1, false>;
    using DummyTauInc_T = DummyGammaDot_t;

    using InternalVariables = std::tuple<FpMap_t, GammaDotMap_t, TauYMap_t, EulerMap_t, DummyGammaDot_t, DummyTauInc_T>;
  };

  /**
   * implements finite strain crystal plasticity
   */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  class MaterialCrystalPlasticityFinite:
    public MaterialMuSpectre<MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>, DimS, DimM>
  {
  public:
    constexpr static Int NbEuler{(DimM==3) ? 3 : 1};

    //! base class
    using Parent = MaterialMuSpectre<MaterialCrystalPlasticityFinite, DimS, DimM>;

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
    using GammaDot_ref = typename traits::GammaDotMap_t::reference;
    //! Type in which CRSS are referenced
    using TauY_ref = typename traits::TauYMap_t::reference;
    //! Type in which Euler angles are referenced
    using Euler_ref = typename traits::EulerMap_t::reference;
    using Dummy_ref = typename traits::DummyTauInc_T::reference;
    //! Type in which slip directions and normals are given
    using SlipVecs = Eigen::Matrix<Real, NbSlip, DimM>;
    using SlipVecs_ref = Eigen::Ref<SlipVecs>;
    //! Type for second rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T2_ref = Eigen::Ref<const T2_t>;
    //! Type for fourth rank tensors
    using T4_t = T4Mat<Real, DimM>;

    //! Default constructor
    MaterialCrystalPlasticityFinite() = delete;

    /**
     * Construct by name, Bulk modulus, Shear modulus, Reference slip
     * rate, Strain rate sensitivity, Initial CRSS, Initial hardening
     * modulus, CRSS saturation value, Hardening exponent,
     * Latent/Self-hardening ratio, Slip directions, Slip normals,
     * Plastic slip rate tolerance, Plastic slip rate loop maximum
     * iterations
     */
    MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m,
                                    Real gamma_dot0, Real m_par, Real tau_y0,
                                    Real h0, Real delta_tau_y_max, Real a_par,
                                    Real q_n, SlipVecs_ref Slip0,
                                    SlipVecs_ref Normal0, Real delta_t,
                                    Real tolerance=1.e-4, Int maxiter=20);

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
    T2_t evaluate_stress(const T2_ref & F, Fp_ref Fp,
                         GammaDot_ref gamma_dot,
                         TauY_ref tau_y, Euler_ref Euler,
                         Dummy_ref dummy_gamma_dot, Dummy_ref dummy_tau_inc);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Deformation Gradient
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_ref & F, Fp_ref Fp, GammaDot_ref gamma_dot,
                            TauY_ref tau_y, Euler_ref Euler,
                         Dummy_ref dummy_gamma_dot, Dummy_ref dummy_tau_inc);

    /**
     * return the internals tuple
     */
    InternalVariables & get_internals() {
      return this->internal_variables;};

    /**
     * overload add_pixel to write into internals
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) override final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel,
                   const Eigen::Ref<Eigen::Array<Real, NbEuler, 1>> Euler);

    /**
     * for introspection
     */
    constexpr static Dim_t get_NbSlip() {return NbSlip;}


    /**
     * set initial values for internal variables
     */
    void initialise() override;

    /**
     * material-specific update of internal (history) variables
     */
    void save_history_variables() override;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  protected:
    using LColl_t = typename traits::LFieldColl_t;
    //! for degenerate case of 1 slip system
    using ScalarPerSlip_map =
      StateFieldMap<MatrixFieldMap<LColl_t, Real, NbSlip, 1>>;
    //! Storage for F_p
    using FpField_t = StateField<TensorField<LColl_t, Real, secondOrder, DimM>>;
    FpField_t & FpField;
    //! Storage for dγ/dt
    using GammaDotField_t = StateField<MatrixField<LColl_t, Real, NbSlip, 1>>;
    GammaDotField_t & GammaDotField;
    //! Storage for τ_y
    using TauYField_t = StateField<MatrixField<LColl_t, Real, NbSlip, 1>>;
    TauYField_t & TauYField;
    //! Storage for γ
    MatrixField<LColl_t, Real, NbSlip, 1> & GammaField;
    //! Storage for Euler angles
    MatrixField<LColl_t, Real, NbEuler, 1> & EulerField;
    //! bulk modulus
    Real bulk_m;
    Real shear_m;
    Real gamma_dot0;
    Real m_par;
    Real tau_y0;
    Real h0;
    Real tau_infty;
    Real a_par;
    Real q_n;
    Real delta_t;
    Real tolerance;
    Int maxiter;
    //! Storage for slip directions
    Eigen::Matrix<Real, NbSlip, DimM> Slip0;
    //! Storage for slip plane normals
    Eigen::Matrix<Real, NbSlip, DimM> Normal0;
    T4_t C_el{};

    //! Storage for dummy γ_dot
    MatrixField<LColl_t, Real, NbSlip, 1> & dummy_gamma_dot;
    //! Storage for dummy τ_inc
    MatrixField<LColl_t, Real, NbSlip, 1> & dummy_tau_inc;

    //! tuple for iterable internal variables
    InternalVariables internal_variables;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  auto
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  evaluate_stress(const T2_ref & F, Fp_ref Fp, GammaDot_ref gamma_dot, TauY_ref tau_y,
                  Euler_ref Euler, Dummy_ref dummy_gamma_dot, Dummy_ref dummy_tau_inc) -> T2_t{
    return std::get<0>
      (this->evaluate_stress_tangent(std::move(F),
                                     Fp, gamma_dot, tau_y,
                                     Euler,
                                     dummy_gamma_dot,
                                     dummy_tau_inc));
  }

}  // muSpectre

#endif /* MATERIAL_CRYSTAL_PLASTICITY_FINITE_H */
