
cmake_minimum_required (VERSION 2.8.11)

project(test_nlopt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# NLOPT options
set(NLOPT_CXX OFF CACHE BOOL "enable cxx routines")
set(NLOPT_PYTHON OFF CACHE BOOL "build python bindings")
set(NLOPT_OCTAVE OFF CACHE BOOL "build octave bindings")
set(NLOPT_MATLAB OFF CACHE BOOL "build matlab bindings")
set(NLOPT_GUILE OFF CACHE BOOL "build guile bindings")
set(NLOPT_SWIG OFF CACHE BOOL "use SWIG to build bindings")
set(NLOPT_LINK_PYTHON OFF CACHE BOOL "link Python libs")

add_subdirectory(nlopt)

# Python wrapper
find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/nlopt/src/api/)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/nlopt)

add_library(wnlopt MODULE wrapper.c)
set_target_properties(wnlopt PROPERTIES PREFIX "")
target_link_libraries(wnlopt nlopt ${PYTHON_LIBRARIES})

# Python script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fit_main.py
  ${CMAKE_CURRENT_BINARY_DIR}/fit_main.py)
