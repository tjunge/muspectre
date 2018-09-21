/**
 * @file   muLib_solver.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   21 Sep 2018
 *
 * @brief  implementation of the mulib solver
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

#include "materials/material_linear_elastic_generic.hh"
#include "solver/mulib_solver.hh"
#include "cell/cell_factory.hh"
#include "solver/solver_cg.hh"
#include "solver/solvers.hh"

namespace muSpectre {

  namespace internal {

    template <Dim_t Dim>
    struct VoigtVec { };

    template <>
    struct VoigtVec<twoD> {
      static decltype(auto) get() {
        Eigen::Matrix<Dim_t, twoD*twoD, 2> mat{};
        mat << 0, 0,
          1, 1,
          0, 1,
          1, 0;
        return mat;
      }
    };
    template <>
    struct VoigtVec<threeD> {
      static decltype(auto) get() {
        Eigen::Matrix<Dim_t, threeD*threeD, 2> mat{};
        mat << 0, 0,
          1, 1,
          2, 2,
          1, 2,
          0, 2,
          0, 1,
          2, 1,
          2, 0,
          1, 0;
        return mat;
      }
    };

  }  // internal

  template <Dim_t Dim>
  void mulib_worker(MuLibInput & file,
                    Real newton_tol, Real equil_tol, Real cg_tol,
                    Uint maxiter, Dim_t verbose) {
    Ccoord_t<Dim> resolutions{};
    for (auto && tup: akantu::zip(resolutions, file.get_resolutions())) {
      std::get<0>(tup) = std::get<1>(tup);
    }
    constexpr Rcoord_t<Dim> lengths{CcoordOps::get_cube<Dim>(1.)};

    constexpr auto form{Formulation::small_strain};

    auto cell {make_cell<Dim, Dim>(resolutions, lengths, form)};
    file.template setup_cell<Dim>(cell);

    std::cout << newton_tol << equil_tol << verbose;

    using Delta_t = Eigen::MatrixXd;
    using Delta_vec = std::vector<Delta_t>;
    Delta_vec Deltas{};
    for (int i{0}; i < Dim; ++i) {
      for (int j{i}; j < Dim; ++j) {
        Eigen::MatrixXd Delta(Dim, Dim);
        Delta.setZero();
        Delta(i, j) +=.5;
        Delta(j, i) +=.5;
        Deltas.push_back(Delta);
      }
    }

    SolverCG cg{cell, cg_tol, maxiter, bool(verbose)};
    auto results = de_geus(cell, Deltas, cg, newton_tol, verbose);


    Delta_vec mean_delta_stresses{};
    T4Mat<Real, Dim> C{};
    C.setZero();
    {
      size_t counter{};
      for (int i{0}; i < Dim; ++i) {
        for (int j{i}; j < Dim; ++j, ++counter) {
          auto get_mean = [](const auto & vec) {
            Eigen::Map<const Eigen::MatrixXd>
            matrix(vec.data(),
                   Dim*Dim, vec.rows()/(Dim*Dim));
            auto& mat{ matrix.rowwise().mean()};
            Eigen::Matrix<Real, Dim, Dim> mean;
            Eigen::Map<Eigen::Matrix<Real, Dim* Dim, 1 >>(mean.data()) = mat;
            return mean;
          };
          const auto & opt_result = results[counter];
          auto stress{get_mean(opt_result.stress)};
          std::cout << std::endl<< "mean stress:" << std::endl
                    << stress << std::endl;
          std::cout << "mean strain:" << std::endl
                    << get_mean(opt_result.grad) << std::endl;

          for (int k{0}; k < Dim; ++k) {
            for (int l{0}; l < Dim; ++l) {
              get(C, i,j,k,l) = get(C, j,i,k,l) = stress(k,l);
            }
          }
        }
      }
    }
    std::cout << "stiffness tensor :" << std::endl
              << C << std::endl;

    Eigen::Matrix<Real, vsize(Dim), vsize(Dim)> C_voigt{};
    C_voigt.setZero();
    const auto redirect{internal::VoigtVec<Dim>::get()};
    for (int I{0}; I < vsize(Dim); ++I) {
      const Dim_t & i{redirect(I,0)};
      const Dim_t & j{redirect(I,1)};
      for (int J{0}; J < vsize(Dim); ++J) {
        const Dim_t & k{redirect(J,0)};
        const Dim_t & l{redirect(J,1)};
        C_voigt(I,J) = get(C, i,j,k,l);
      }
    }
    std::cout << "stiffness tensor voigt :" << std::endl
              << C_voigt << std::endl;

  }


  void mulib(const filesystem::path & path,
             Real newton_tol, Real equil_tol, Real cg_tol,
             Uint maxiter, Dim_t verbose) {
    MuLibInput file{path};
    const auto dim{file.get_dim()};

    switch (dim) {
    case twoD: {
      return mulib_worker<twoD>(file, newton_tol, equil_tol, cg_tol,
                                maxiter, verbose);
      break;
    }
    case threeD: {
      return mulib_worker<threeD>(file, newton_tol, equil_tol, cg_tol,
                                  maxiter, verbose);
      break;
    }
    default:
      throw std::runtime_error("only two- and tree-dimensional cases considered");
      break;
    }
  }

}  // muSpectre
