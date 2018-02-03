#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   01 Feb 2018
#
# @brief  builds the µSpectre documentation
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

find_package(Sphinx REQUIRED)

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/conf.py"
  "${CMAKE_CURRENT_BINARY_DIR}/conf.py"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/Doxyfile"
  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/input_def.in"
  "${CMAKE_CURRENT_BINARY_DIR}/input_def"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/xml_output_def.in"
  "${CMAKE_CURRENT_BINARY_DIR}/xml_output_def"
  @ONLY)

add_custom_target(muSpectre_docs
  ${SPHINX_EXECUTABLE}
  -q -b html
  -c "${CMAKE_CURRENT_BINARY_DIR}"
  -d "${SPHINX_CACHE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source"
  "${SPHINX_HTML_DIR}"
  COMMENT "Building HTML documentation with Sphinx")