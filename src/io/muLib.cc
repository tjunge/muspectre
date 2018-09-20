/**
 * file   muLib.cc
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


#include "io/muLib.hh"

#include <ncException.h>

#include <sstream>
#include <system_error>

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

    auto mat_indices{this->file.getVar("materialIndices")};
    if (mat_indices.isNull()) {
      throw std::runtime_error("Could not read material indices");
    }

    auto dims{mat_indices.getDims()};

    for (auto & dim: dims) {
      std::cout << dim.getName() << std::endl;
    }

  }

}  // muSpectre
