# CMakeLists.txt for building the C library (liboptv)
# Note: Python bindings are now built using setuptools (see bindings/setup.py)
# This CMakeLists.txt is kept for C-only projects that want to use liboptv directly

cmake_minimum_required(VERSION 3.15)
project(optv VERSION 1.0.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build the C library
set(LIBOPTV_SOURCES
    src/tracking_frame_buf.c
    src/calibration.c
    src/parameters.c
    src/lsqadj.c
    src/ray_tracing.c
    src/trafo.c
    src/vec_utils.c
    src/image_processing.c
    src/multimed.c
    src/imgcoord.c
    src/epi.c
    src/orientation.c
    src/sortgrid.c
    src/segmentation.c
    src/correspondences.c
    src/track.c
    src/tracking_run.c
    src/track3d.c
)

# Create the C library (static)
add_library(optv_lib STATIC ${LIBOPTV_SOURCES})
target_include_directories(optv_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(UNIX)
    target_link_libraries(optv_lib PRIVATE m)
endif()

# Install the library and headers
install(TARGETS optv_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Option to build tests
option(BUILD_TESTS "Build C library tests" OFF)
if(BUILD_TESTS)
    find_package(Check REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()
