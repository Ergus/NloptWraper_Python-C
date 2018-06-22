#
# This file is part of the NloptWraper_Python-C distribution Copyright (c) 2017
# Jimmy Aguilar Mena.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required (VERSION 2.8.11)

project(test_nlopt)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_C_FLAGS_DEBUG "-g -pg -no-pie") 

# NLOPT options
set(NLOPT_CXX OFF CACHE BOOL "enable cxx routines")
set(NLOPT_PYTHON OFF CACHE BOOL "build python bindings")
set(NLOPT_OCTAVE OFF CACHE BOOL "build octave bindings")
set(NLOPT_MATLAB OFF CACHE BOOL "build matlab bindings")
set(NLOPT_GUILE OFF CACHE BOOL "build guile bindings")
set(NLOPT_SWIG OFF CACHE BOOL "use SWIG to build bindings")
set(NLOPT_LINK_PYTHON OFF CACHE BOOL "link Python libs")

add_subdirectory(nlopt)

#%rename(NLOPT_GN_DIRECT) nlopt::GN_DIRECT;

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nlopt-enum.h)
  message("Generating header on the fly")
  file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/nlopt-enum.h
	"// AUTOMATICALLY GENERATED -- DO NOT EDIT\n")
    file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/nlopt-enum.h
	"#define ADDVALUES(MODULE) { \\\n")
  file (STRINGS ${PROJECT_SOURCE_DIR}/nlopt/src/api/nlopt.h
	NLOPT_H_LINES REGEX "    NLOPT_[A-Z0-9_]+")
  foreach (NLOPT_H_LINE ${NLOPT_H_LINES})
    #string (REGEX REPLACE ".*NLOPT_([A-Z0-9_]+).*" "\tNLOPT_\\1 ,\n"
	string (REGEX REPLACE ".*NLOPT_([A-Z0-9_]+).*"
	  "\tPyModule_AddIntConstant(MODULE, \"NLOPT_\\1\", NLOPT_\\1);\\\\\n"
	  ENUM_LINE ${NLOPT_H_LINE})
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/nlopt-enum.h "${ENUM_LINE}")
  endforeach ()
  file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/nlopt-enum.h "}\n")
endif ()

# Python wrapper
find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/nlopt/src/api/
  ${CMAKE_CURRENT_BINARY_DIR})
link_directories(${CMAKE_CURRENT_BINARY_DIR}/nlopt)

add_library(wnlopt MODULE wrapper.c)
set_target_properties(wnlopt PROPERTIES PREFIX "")
target_link_libraries(wnlopt nlopt ${PYTHON_LIBRARIES})

# C benchmark
add_executable(fit_main fit_main.c)
target_link_libraries(fit_main nlopt)

# Python script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fit_main.py
  ${CMAKE_CURRENT_BINARY_DIR}/fit_main.py)
