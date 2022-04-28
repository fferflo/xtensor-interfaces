get_filename_component(xtensor-interfaces_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET xtensor-interfaces::base)
  find_package(xtl REQUIRED)
  find_package(xtensor REQUIRED)
  find_package(xtensor-blas REQUIRED)

  find_package(OpenCV QUIET)
  find_package(Eigen3 QUIET)

  include("${xtensor-interfaces_CMAKE_DIR}/xtensor-interfacesTargets.cmake")
endif()
