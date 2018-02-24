#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  configuration for python binding using pybind11
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

if(APPLE)
  # pybind11 (installed via homebrew) does not set include directories properly.
  # This is necessary to find Python.h.
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
      "from distutils import sysconfig as s;print(s.get_python_inc(plat_specific=True));"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUES
    ERROR_VARIABLE _PYTHON_ERROR_VALUE)
  list(GET _PYTHON_VALUES 0 PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
endif()

set (PY_BINDING_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_module.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_system.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_material.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_solvers.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_fftengine.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_projections.cc
  )

pybind11_add_module(pyMuSpectre ${PY_BINDING_SRCS})
target_link_libraries(pyMuSpectre PRIVATE muSpectre)
# Want to rename the output, so that the python module is called muSpectre
set_target_properties(pyMuSpectre PROPERTIES OUTPUT_NAME muSpectre)