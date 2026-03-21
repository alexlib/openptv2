cmake_minimum_required(VERSION 3.15)
project(openptv2 VERSION 1.0.0 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard for Cython-generated code
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Set RPATH for shared libraries
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")

# Find Python and required packages
find_package(Python 3.11 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Get NumPy include directory
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build the C library
set(LIBOPTV_SOURCES
    lib/src/tracking_frame_buf.c
    lib/src/calibration.c
    lib/src/parameters.c
    lib/src/lsqadj.c
    lib/src/ray_tracing.c
    lib/src/trafo.c
    lib/src/vec_utils.c
    lib/src/image_processing.c
    lib/src/multimed.c
    lib/src/imgcoord.c
    lib/src/epi.c
    lib/src/orientation.c
    lib/src/sortgrid.c
    lib/src/segmentation.c
    lib/src/correspondences.c
    lib/src/track.c
    lib/src/tracking_run.c
    lib/src/track3d.c
)

# Create the C library
add_library(optv STATIC ${LIBOPTV_SOURCES})
target_include_directories(optv PUBLIC ${CMAKE_SOURCE_DIR}/lib/include)

if(UNIX)
    target_link_libraries(optv PRIVATE m)
endif()

# Build Python bindings if enabled
if(BUILD_PYTHON_BINDINGS)
    # Find Cython
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m cython --version
        OUTPUT_VARIABLE CYTHON_VERSION_OUTPUT
        ERROR_VARIABLE CYTHON_VERSION_OUTPUT
        RESULT_VARIABLE CYTHON_RESULT
    )
    
    if(CYTHON_RESULT EQUAL 0)
        message(STATUS "Cython found")
        
        # Create output directory for optv package
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/optv")
        
        # Cython sources
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
            set(PYX_FILE "${CMAKE_SOURCE_DIR}/bindings/optv/${MODULE}.pyx")
            set(PXD_FILE "${CMAKE_SOURCE_DIR}/bindings/optv/${MODULE}.pxd")
            set(C_FILE "${CMAKE_CURRENT_BINARY_DIR}/optv/${MODULE}.cxx")
            
            # Run Cython to generate C++ file in build directory
            add_custom_command(
                OUTPUT ${C_FILE}
                COMMAND ${Python_EXECUTABLE} -m cython --cplus -3 
                    -I ${CMAKE_SOURCE_DIR}/bindings
                    -I ${CMAKE_SOURCE_DIR}/bindings/optv
                    -o ${C_FILE}
                    ${PYX_FILE}
                DEPENDS ${PYX_FILE} ${PXD_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running Cython on ${MODULE}.pyx"
            )
            
            # Create Python extension module using CXX compiler
            add_library(optv_${MODULE} MODULE ${C_FILE})
            
            target_include_directories(optv_${MODULE} PRIVATE
                ${CMAKE_SOURCE_DIR}/bindings
                ${CMAKE_SOURCE_DIR}/bindings/optv
                ${CMAKE_SOURCE_DIR}/lib/include
                ${NUMPY_INCLUDE_DIR}
            )
            
            target_link_libraries(optv_${MODULE} PRIVATE Python::Module optv)
            
            set_target_properties(optv_${MODULE} PROPERTIES
                PREFIX ""
                OUTPUT_NAME "${MODULE}"
                SUFFIX "${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/optv"
            )
            
            # Install to optv package directory
            install(TARGETS optv_${MODULE}
                LIBRARY DESTINATION optv
            )
        endforeach()
        
        # Install Python package files for optv (only .py and .pxd files)
        install(DIRECTORY bindings/optv/
            DESTINATION optv
            FILES_MATCHING PATTERN "*.py"
            PATTERN "*.pxd"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.c" EXCLUDE
            PATTERN "*.cxx" EXCLUDE
        )
    else()
        message(WARNING "Cython not found, skipping Python bindings")
    endif()
endif()

# Install Python package files for openptv2
install(DIRECTORY openptv2/
    DESTINATION openptv2
    FILES_MATCHING PATTERN "*.py"
    PATTERN "*.pxd"
    PATTERN "__pycache__" EXCLUDE
)

# Install GUI files
install(DIRECTORY gui/pyptv/
    DESTINATION pyptv
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)

install(DIRECTORY gui/pyptv/calibration_output/
    DESTINATION pyptv/calibration_output
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)

# Install headers
install(DIRECTORY lib/include/
    DESTINATION include/openptv2
    FILES_MATCHING PATTERN "*.h"
)

# Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(lib/tests)
endif()

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "openptv2")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_CONTACT "OpenPTV Community <openptv@googlegroups.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Unified OpenPTV: Particle Tracking Velocimetry")
include(CPack)
