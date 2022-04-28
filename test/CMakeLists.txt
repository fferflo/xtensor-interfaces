######################## ACQUIRE TEST DATA ########################

set(AIRPLANE_TIFF_URL "http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.05")
set(AIRPLANE_TIFF "${CMAKE_BINARY_DIR}/data/airplane.tiff")
set(AIRPLANE_TIFF_WIDTH 512)
set(AIRPLANE_TIFF_HEIGHT 512)
ExternalProject_Add(
    airplane_tiff
    PREFIX "tmp"
    DOWNLOAD_NAME "airplane.tiff"
    DOWNLOAD_NO_EXTRACT ON
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/data"
    URL ${AIRPLANE_TIFF_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)

set(AIRPLANE_PNG "${CMAKE_BINARY_DIR}/data/airplane.png")
set(AIRPLANE_PNG_WIDTH 512)
set(AIRPLANE_PNG_HEIGHT 512)
add_custom_target(airplane_png
  COMMAND convert ${AIRPLANE_TIFF} ${AIRPLANE_PNG}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  VERBATIM
)
add_dependencies(airplane_png airplane_tiff)

set(AIRPLANE_HALF_PNG "${CMAKE_BINARY_DIR}/data/airplane_half.png")
set(AIRPLANE_HALF_PNG_WIDTH 512)
set(AIRPLANE_HALF_PNG_HEIGHT 256)
add_custom_target(airplane_half_png
  COMMAND convert ${AIRPLANE_PNG} -crop 512x256+0+0 ${AIRPLANE_HALF_PNG}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  VERBATIM
)
add_dependencies(airplane_half_png airplane_png)

######################## TESTS ########################

add_custom_target(tests)
find_package(Catch2 3 REQUIRED)

macro(xti_add_test TARGET TEST_NAME)
  add_executable(${TARGET} EXCLUDE_FROM_ALL ${TEST_NAME})
  target_link_libraries(${TARGET} base Catch2::Catch2WithMain)
  add_test(NAME ${TARGET} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test" COMMAND "${CMAKE_BINARY_DIR}/bin/test/${TARGET}")
  set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
  add_dependencies(tests ${TARGET})
endmacro()

if(TARGET opencv)
  xti_add_test(test_opencv opencv.cpp)
  target_link_libraries(test_opencv opencv)
  add_dependencies(test_opencv airplane_half_png)
  target_compile_definitions(test_opencv PRIVATE
    IMAGE_PATH="${AIRPLANE_HALF_PNG}"
    IMAGE_WIDTH=${AIRPLANE_HALF_PNG_WIDTH}
    IMAGE_HEIGHT=${AIRPLANE_HALF_PNG_HEIGHT}
  )
endif()

if(TARGET eigen)
  xti_add_test(test_eigen eigen.cpp)
  target_link_libraries(test_eigen eigen)
endif()

if(TARGET cereal)
  xti_add_test(test_cereal cereal.cpp)
  target_link_libraries(test_cereal cereal)
endif()
