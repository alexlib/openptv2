# CMakeLists.txt for building Cython bindings with scikit-build-core
cmake_minimum_required(VERSION 3.15)
project(openptv2_bindings VERSION 1.0.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Python and required packages
find_package(Python 3.11 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Find Cython
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import cython; print(cython.get_include())"
    OUTPUT_VARIABLE CYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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
add_library(optv STATIC ${LIBOPTV_SOURCES})
target_include_directories(optv PUBLIC ../lib/include)

if(UNIX)
    target_link_libraries(optv PRIVATE m)
endif()

# Cython sources in bindings/optv/
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
    set(C_FILE "optv/${MODULE}.c")
    
    # Run Cython to generate C file
    add_custom_command(
        OUTPUT ${C_FILE}
        COMMAND ${Python_EXECUTABLE} -m cython --cplus -3 -I ${CMAKE_CURRENT_SOURCE_DIR} ${PYX_FILE}
        DEPENDS ${PYX_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running Cython on ${MODULE}.pyx"
    )
    
    # Create Python extension module
    Python_add_library(optv_${MODULE} MODULE ${C_FILE})
    
    target_include_directories(optv_${MODULE} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ../lib/include
        ${NUMPY_INCLUDE_DIR}
    )
    
    target_link_libraries(optv_${MODULE} PRIVATE Python::Module optv)
    
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
    PATTERN "__pycache__" EXCLUDE
)
