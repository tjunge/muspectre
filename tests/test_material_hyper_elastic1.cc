/**
 * file   test_material_hyper_elastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   28 Nov 2017
 *
 * @brief  Tests for the large-strain, objective Hooke's law, implemented in
 *         the convenient strategy (i.e., using MaterialMuSpectre)
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
#include <type_traits>

#include <boost/mpl/list.hpp>

#include "materials/material_hyper_elastic1.hh"
#include "tests.hh"
#include "common/test_goodies.hh"
#include "common/field_collection.hh"
#include "common/iterators.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elastic_1);

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialFixture
  {
    using Mat_t = MaterialHyperElastic1<DimS, DimM>;
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real young{mu*(3*lambda + 2*mu)/(lambda + mu)};
    constexpr static Real poisson{lambda/(2*(lambda + mu))};
    MaterialFixture():mat("Name", young, poisson){};
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};

    Mat_t mat;
  };

  using mat_list = boost::mpl::list<MaterialFixture<twoD, twoD>,
                                    MaterialFixture<twoD, threeD>,
                                    MaterialFixture<threeD, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.mdim());
  }


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    constexpr Dim_t sdim{Fix::sdim};
    testGoodies::RandRange<size_t> rng;;
    const Dim_t nb_pixel{7}, box_size{17};
    using Ccoord = Ccoord_t<sdim>;
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      Ccoord c;
      for (Dim_t j = 0; j < sdim; ++j) {
        c[j] = rng.randval(0, box_size);
      }
      BOOST_CHECK_NO_THROW(mat.add_pixel(c));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialFixtureFilled: public MaterialFixture<DimS, DimM>
  {
    using Mat_t = typename MaterialFixture<DimS, DimM>::Mat_t;
    constexpr static Dim_t box_size{3};
    MaterialFixtureFilled():MaterialFixture<DimS, DimM>(){
      using Ccoord = Ccoord_t<DimS>;
      Ccoord cube{CcoordOps::get_cube<DimS>(box_size)};
      CcoordOps::Pixels<DimS> pixels(cube);
      for (auto pixel: pixels) {
        this->mat.add_pixel(pixel);
      }
      this->mat.initialise();
    };
  };

  using mat_fill = boost::mpl::list<MaterialFixtureFilled<twoD, twoD>,
                                    MaterialFixtureFilled<twoD, threeD>,
                                    MaterialFixtureFilled<threeD, threeD>>;
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_law, Fix, mat_fill, Fix) {
    constexpr auto cube{CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    auto & mat{Fix::mat};

    using FC_t = FieldCollection<Fix::sdim, Fix::mdim>;
    FC_t globalfields;
    auto & F{make_field<typename Fix::Mat_t::StrainField_t>
        ("Transformation Gradient", globalfields)};
    auto & P1 = make_field<typename Fix::Mat_t::StressField_t>
      ("Nominal Stress1", globalfields); // to be computed alone
    auto & P2 = make_field<typename Fix::Mat_t::StressField_t>
      ("Nominal Stress2", globalfields); // to be computed with tangent
    auto & K = make_field<typename Fix::Mat_t::TangentField_t>
      ("Tangent Moduli", globalfields); // to be computed with tangent
    auto & Pr = make_field<typename Fix::Mat_t::StressField_t>
      ("Nominal Stress reference", globalfields);
    auto & Kr = make_field<typename Fix::Mat_t::TangentField_t>
      ("Tangent Moduli reference", globalfields); // to be computed with tangent

    globalfields.initialise(cube);

    static_assert(std::is_same<decltype(P1),
                  typename Fix::Mat_t::StressField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(F),
                  typename Fix::Mat_t::StrainField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(P1), decltype(P2)&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(K),
                  typename Fix::Mat_t::TangentField_t&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(Pr), decltype(P1)&>::value,
                  "oh oh");
    static_assert(std::is_same<decltype(Kr), decltype(K)&>::value,
                  "oh oh");

    { // block to contain not-constant gradient map
      typename Fix::Mat_t::StressMap_t grad_map
        (globalfields["Transformation Gradient"]);
      for (auto F_: grad_map) {
        F_.setRandom();
      }
      grad_map[0] = grad_map[0].Identity(); // identifiable gradients for debug
      grad_map[1] = 1.2*grad_map[1].Identity(); // ditto
    }

    //compute stresses using material
    mat.compute_stresses(globalfields["Transformation Gradient"],
                         globalfields["Nominal Stress1"],
                         Formulation::finite_strain);

    //compute stresses and tangent moduli using material
    BOOST_CHECK_THROW
      (mat.compute_stresses_tangent(globalfields["Transformation Gradient"],
                                    globalfields["Nominal Stress2"],
                                    globalfields["Nominal Stress2"],
                                    Formulation::finite_strain),
       std::runtime_error);

    mat.compute_stresses_tangent(globalfields["Transformation Gradient"],
                                 globalfields["Nominal Stress2"],
                                 globalfields["Tangent Moduli"],
                                 Formulation::finite_strain);

    typename Fix::Mat_t::StrainMap_t Fmap(globalfields["Transformation Gradient"]);
    typename Fix::Mat_t::StressMap_t Pmap_ref(globalfields["Nominal Stress reference"]);
    typename Fix::Mat_t::TangentMap_t Kmap_ref(globalfields["Tangent Moduli reference"]);

    for (auto tup: akantu::zip(Fmap, Pmap_ref, Kmap_ref)) {
      auto F_ = std::get<0>(tup);
      auto P_ = std::get<1>(tup);
      auto K_ = std::get<2>(tup);
      std::tie(P_,K_) = testGoodies::objective_hooke_explicit<Fix::mdim>
        (Fix::lambda, Fix::mu, F_);
    }

    typename Fix::Mat_t::StressMap_t Pmap_1(globalfields["Nominal Stress1"]);
    for (auto tup: akantu::zip(Pmap_ref, Pmap_1)) {
      auto P_r = std::get<0>(tup);
      auto P_1 = std::get<1>(tup);
      Real error = (P_r - P_1).norm();
      BOOST_CHECK_LT(error, tol);
    }

    typename Fix::Mat_t::StressMap_t Pmap_2(globalfields["Nominal Stress2"]);
    typename Fix::Mat_t::TangentMap_t Kmap(globalfields["Tangent Moduli"]);
    for (auto tup: akantu::zip(Pmap_ref, Pmap_2, Kmap_ref, Kmap)) {
      auto P_r = std::get<0>(tup);
      auto P = std::get<1>(tup);
      Real error = (P_r - P).norm();
      std::cout << "P_r =" << std::endl << P_r << std::endl;
      std::cout << "P =" << std::endl << P << std::endl;
      BOOST_CHECK_LT(error, tol);

      auto K_r = std::get<2>(tup);
      auto K = std::get<3>(tup);
      std::cout << "K_r =" << std::endl << K_r << std::endl;
      std::cout << "K =" << std::endl << K << std::endl;
      error = (K_r - K).norm();
      BOOST_CHECK_LT(error, tol);
    }

  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre