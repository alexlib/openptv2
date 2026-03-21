# CMakeLists.txt for building Cython bindings with scikit-build-core
cmake_minimum_required(VERSION 3.15)
project(optv VERSION 1.0.0 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard for Cython-generated code
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and required packages
find_package(Python 3.11 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Get NumPy include directory
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build the C library first
set(LIBOPTV_SOURCES
    ../lib/src/tracking_frame_buf.c
    ../lib/src/calibration.c
    ../lib/src/parameters.c
    ../lib/src/lsqadj.c
    ../lib/src/ray_tracing.c
    ../lib/src/trafo.c
    ../lib/src/vec_utils.c
    ../lib/src/image_processing.c
    ../lib/src/multimed.c
    ../lib/src/imgcoord.c
    ../lib/src/epi.c
    ../lib/src/orientation.c
    ../lib/src/sortgrid.c
    ../lib/src/segmentation.c
    ../lib/src/correspondences.c
    ../lib/src/track.c
    ../lib/src/tracking_run.c
    ../lib/src/track3d.c
)

# Create the C library
add_library(optv_lib STATIC ${LIBOPTV_SOURCES})
target_include_directories(optv_lib PUBLIC ${CMAKE_SOURCE_DIR}/../lib/include)

if(UNIX)
    target_link_libraries(optv_lib PRIVATE m)
endif()

# Cython sources in optv/
set(CYTHON_MODULES
    tracking_framebuf
    tracker
    calibration
    correspondences
    epipolar
    image_processing
    imgcoord
    orientation
    parameters
    segmentation
    transforms
)

# Build each Cython module
foreach(MODULE ${CYTHON_MODULES})
    set(PYX_FILE "optv/${MODULE}.pyx")
    set(PXD_FILE "optv/${MODULE}.pxd")
    set(C_FILE "${CMAKE_CURRENT_BINARY_DIR}/optv/${MODULE}.cxx")

    # Run Cython to generate C++ file
    add_custom_command(
        OUTPUT ${C_FILE}
        COMMAND ${Python_EXECUTABLE} -m cython --cplus -3
            -I ${CMAKE_SOURCE_DIR}
            -I ${CMAKE_SOURCE_DIR}/optv
            -o ${C_FILE}
            ${PYX_FILE}
        DEPENDS ${PYX_FILE} ${PXD_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running Cython on ${MODULE}.pyx"
    )

    # Create Python extension module using CXX compiler
    Python_add_library(optv_${MODULE} MODULE ${C_FILE})

    target_include_directories(optv_${MODULE} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/optv
        ${CMAKE_SOURCE_DIR}/../lib/include
        ${NUMPY_INCLUDE_DIR}
    )

    target_link_libraries(optv_${MODULE} PRIVATE Python::Module optv_lib)

    set_target_properties(optv_${MODULE} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${MODULE}"
        SUFFIX "${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    )

    # Install to optv package directory
    install(TARGETS optv_${MODULE}
        LIBRARY DESTINATION optv
    )
endforeach()

# Install Python package files
install(DIRECTORY optv/
    DESTINATION optv
    FILES_MATCHING PATTERN "*.py"
    PATTERN "*.pxd"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.c" EXCLUDE
    PATTERN "*.cxx" EXCLUDE
)
