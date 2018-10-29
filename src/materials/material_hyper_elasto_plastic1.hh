/**
 * @file   material_hyper_elasto_plastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Feb 2018
 *
 * @brief  Material for logarithmic hyperelasto-plasticity, as defined in de
 *         Geus 2017 (https://doi.org/10.1016/j.cma.2016.12.032) and further
 *         explained in Geers 2003 (https://doi.org/10.1016/j.cma.2003.07.014)
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */



#ifndef MATERIAL_HYPER_ELASTO_PLASTIC1_H
#define MATERIAL_HYPER_ELASTO_PLASTIC1_H

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/eigen_tools.hh"
#include "common/statefield.hh"

#include <algorithm>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialHyperElastoPlastic1;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<DimS, DimM>> {
    //! global field collection
    using GFieldCollection_t = typename
      MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
      MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};

    //! local field collection used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;

    //! storage type for plastic flow measure (εₚ in the papers)
    using LScalarMap_t = StateFieldMap<ScalarFieldMap<LFieldColl_t, Real>>;
    /**
     * storage type for for previous gradient Fᵗ and elastic left
     * Cauchy-Green deformation tensor bₑᵗ
     */
    using LStrainMap_t = StateFieldMap<
      MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, false>>;

    /**
     * format in which to receive internals (previous gradient Fᵗ,
     * previous elastic lef Cauchy-Green deformation tensor bₑᵗ, and
     * the plastic flow measure εₚ
     */
    using InternalVariables = std::tuple<LStrainMap_t, LStrainMap_t,
                                         LScalarMap_t>;

  };


  /**
   * material implementation for hyper-elastoplastic constitutive law
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class MaterialHyperElastoPlastic1: public
    MaterialMuSpectre<MaterialHyperElastoPlastic1<DimS, DimM>, DimS, DimM>
  {
  public:

    //! base class
    using Parent = MaterialMuSpectre
      <MaterialHyperElastoPlastic1<DimS, DimM>, DimS, DimM>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = T4Mat<Real, DimM>;

    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;

    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1>;

    //! Hooke's law implementation
    using Hooke = typename
      MatTB::Hooke<DimM,
                   typename traits::StrainMap_t::reference,
                   typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using StrainStRef_t = typename traits::LStrainMap_t::reference;
    //! type in which the previous plastic flow is referenced
    using FlowStRef_t = typename traits::LScalarMap_t::reference;

    //! Default constructor
    MaterialHyperElastoPlastic1() = delete;

    //! Constructor with name and material properties
    MaterialHyperElastoPlastic1(std::string name, Real young, Real poisson,
                                Real tau_y0, Real H);

    //! Copy constructor
    MaterialHyperElastoPlastic1(const MaterialHyperElastoPlastic1 &other) = delete;

    //! Move constructor
    MaterialHyperElastoPlastic1(MaterialHyperElastoPlastic1 &&other) = delete;

    //! Destructor
    virtual ~MaterialHyperElastoPlastic1() = default;

    //! Copy assignment operator
    MaterialHyperElastoPlastic1& operator=(const MaterialHyperElastoPlastic1 &other) = delete;

    //! Move assignment operator
    MaterialHyperElastoPlastic1& operator=(MaterialHyperElastoPlastic1 &&other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    T2_t evaluate_stress(const T2_t& F, StrainStRef_t F_prev,
                         StrainStRef_t be_prev,
                         FlowStRef_t plast_flow);
    /**
     * evaluates Kirchhoff stress and stiffness given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    std::tuple<T2_t, T4_t> evaluate_stress_tangent(const T2_t & F,
                                                   StrainStRef_t F_prev,
                                                   StrainStRef_t be_prev,
                                                   FlowStRef_t plast_flow);

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    virtual void save_history_variables() override;

    /**
     * set the previous gradients to identity
     */
    virtual void initialise() override final;

    /**
     * return the internals tuple
     */
    typename traits::InternalVariables & get_internals() {
      return this->internal_variables;};


  protected:

    /**
     * worker function computing stresses and internal variables
     */
    using Worker_t = std::tuple<T2_t, Real, Real, T2_t, bool, Decomp_t<DimM>>;
    Worker_t stress_n_internals_worker(const T2_t & F,
                                       StrainStRef_t& F_prev,
                                       StrainStRef_t& be_prev,
                                       FlowStRef_t& plast_flow);
    //! Local FieldCollection type for field storage
    using LColl_t = LocalFieldCollection<DimS>;
    //! storage for cumulated plastic flow εₚ
    StateField<ScalarField<LColl_t, Real>> & plast_flow_field;

    //! storage for previous gradient Fᵗ
    StateField<TensorField<LColl_t, Real, secondOrder, DimM>> & F_prev_field;

    //! storage for elastic left Cauchy-Green deformation tensor bₑᵗ
    StateField<TensorField<LColl_t, Real, secondOrder, DimM>> & be_prev_field;

    // material properties
    const Real young;          //!< Young's modulus
    const Real poisson;        //!< Poisson's ratio
    const Real lambda;         //!< first Lamé constant
    const Real mu;             //!< second Lamé constant (shear modulus)
    const Real K;              //!< Bulk modulus
    const Real tau_y0;         //!< initial yield stress
    const Real H;              //!< hardening modulus
    const T4Mat<Real, DimM> C; //!< stiffness tensor

    //! Field maps and state field maps over internal fields
    typename traits::InternalVariables internal_variables;
  private:
  };


}  // muSpectre

#endif /* MATERIAL_HYPER_ELASTO_PLASTIC1_H */
