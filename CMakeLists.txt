cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(xtensor-interfaces LANGUAGES CXX)

enable_testing()

include(GNUInstallDirs)
include(ExternalProject)

set(PACKAGE_NAME xtensor-interfaces)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME})


# Base library
add_library(base INTERFACE)
target_include_directories(base
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(base INTERFACE cxx_std_17)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
target_link_libraries(base INTERFACE xtensor)

install(
  TARGETS base
  EXPORT install_targets
)

######################## INTERFACES ########################

# OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  add_library(opencv INTERFACE)
  target_link_libraries(opencv INTERFACE base ${OpenCV_LIBS})
  target_include_directories(opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
  install(
    TARGETS opencv
    EXPORT install_targets
  )
endif()

# Eigen
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  add_library(eigen INTERFACE)
  target_link_libraries(eigen INTERFACE base Eigen3::Eigen)
  install(
    TARGETS eigen
    EXPORT install_targets
  )
endif()

# Cereal
find_package(cereal QUIET)
if(cereal_FOUND)
  add_library(cereal INTERFACE)
  target_link_libraries(cereal INTERFACE base cereal::cereal)
  install(
    TARGETS cereal
    EXPORT install_targets
  )
endif()


######################## TESTS ########################

add_subdirectory(test)



######################## INSTALL ########################

# Install headers
set(INSTALL_HEADERS_AS_SYMLINK OFF CACHE BOOL "Installs headers as a symlink to the source directory instead of copying all files to install directory")
if(INSTALL_HEADERS_AS_SYMLINK)
  install(CODE "file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})")
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include/xti ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/xti)")
  install(CODE "message(\"-- Installed symlink: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/xti -> ${CMAKE_CURRENT_SOURCE_DIR}/include/xti\")")
else()
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Install targets file
install(EXPORT install_targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    ${PACKAGE_NAME}::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)
# Install ${PACKAGE_NAME}Config.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
