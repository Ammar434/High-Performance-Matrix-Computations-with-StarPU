cmake_minimum_required(VERSION 3.12)
cmake_policy(VERSION 3.12)
project(gemm LANGUAGES C CXX)

# Define the compilation options of the project
# ---------------------------------------------

# Option to enable/disable the MPI compilation
option(ENABLE_MPI "Define if MPI support should be enabled" OFF)

# Option to enable/disable CUDA support
option(ENABLE_CUDA "Define if CUDA support is enabled or not" OFF)

#Option to enable/disable OpenMP support for parallel workers
option(STARPU_OPENMP "Define if OpenMP support is enabled or not" OFF)

if(${ENABLE_CUDA})
	enable_language(CUDA)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

add_compile_options(
	-Wall -Wextra # base warnings
	-Wfloat-equal # prevent comparing floats with ==
	-Wconversion -Wsign-conversion # type conversion
	-Wformat=2 # check printf-style templates string
	-Wpointer-arith # warn on sizeof(fn) or sizeof(void) (enabled by Wpedantic but we can't use that because of nvcc)
	-Wcast-align
	-Wno-c++20-extensions -Wno-missing-field-initializers -Wno-unused-parameter
)

# Check that we do no try to configure/build inside the source directory
# ----------------------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "
		This project is set such that you can not configure/build inside the source directory.
		Please:
		 1) remove entirely this directory
		 2) extract or clone a fresh revision of the project
		 3) create a build directory and run cmake from this directory or
		run with (for exemple):
		mkdir build && cd build && cmake ..
		or
		cmake . -B./build
")
endif()

# Disable in source directory build
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# use, i.e. don't skip, the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Build shared libraries
set(BUILD_SHARED_LIBS ON)

# Check that the submodule exists and is up to date
# -------------------------------------------------
find_package(Git)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
	# Update submodules as needed
	option(GIT_SUBMODULE "Check submodules during build" ON)
	if(GIT_SUBMODULE)
		message(STATUS "Submodule update")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
		endif()
	endif()
endif()

if( NOT EXISTS "${PROJECT_SOURCE_DIR}/submodules/morse_cmake/modules/MorseInit.cmake" )
	message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Add extra cmake module path and initialize morse cmake modules
# This is mainly done to integrate the BLAS/LAPACK detection
# --------------------------------------------------------------
if (IS_DIRECTORY ${CMAKE_SOURCE_DIR}/submodules/morse_cmake/modules)
	set( MORSE_CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/submodules/morse_cmake/modules )
	list(APPEND CMAKE_MODULE_PATH ${MORSE_CMAKE_MODULE_PATH})
	include(MorseInit)
else()
	message(FATAL_ERROR "Submodule submodules/morse_cmake not initialized - run `git submodule update --init`")
endif()

# Find all required external packages
# -----------------------------------
# Search for MPI
if (ENABLE_MPI)
	add_compile_definitions(USE_MPI)
	find_package(MPI REQUIRED)
endif()

if(${ENABLE_CUDA})
	# enable CUDA
	add_compile_definitions(USE_CUDA)
	find_package(CUDAToolkit)

	# check whether CUDA is available
	include(CheckLanguage)
	check_language(CUDA)
endif()

# Search for StarPU
# Specify the minimum version
set(STARPU_REQUIRED_VERSION "1.4" CACHE STRING "Newer STARPU version required")
find_package(STARPU ${STARPU_REQUIRED_VERSION} REQUIRED)
morse_cmake_required_set(STARPU)

# Search for OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    if(${STARPU_OPENMP})
        add_compile_definitions(STARPU_OPENMP)
    endif()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Search for libm
find_package(M REQUIRED)

# Search for cblas in sequential
set(CBLAS_MT FALSE)
find_package(CBLAS REQUIRED)

include(src/sources.cmake)

if(${ENABLE_CUDA})
	list(APPEND cpp_sources ${cuda_sources})
endif()

# Create library and link libraries
add_library(gemmlib ${cpp_sources})

if(${ENABLE_MPI})
	target_link_libraries(gemmlib PUBLIC MPI::MPI_CXX)
endif()

if(${ENABLE_CUDA})
	set_property(TARGET gemmlib PROPERTY CUDA_ARCHITECTURES OFF)
	target_link_libraries(gemmlib PUBLIC CUDA::cublas)
	target_link_libraries(gemmlib PUBLIC CUDA::cudart)
endif()

target_link_libraries(gemmlib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(gemmlib PUBLIC MORSE::CBLAS)
target_link_libraries(gemmlib PUBLIC MORSE::M)
target_link_libraries(gemmlib PUBLIC MORSE::STARPU)

install(TARGETS gemmlib
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib)

# Add executable
add_executable(gemm src/gemm.cpp)
target_link_libraries(gemm PRIVATE gemmlib)
install(TARGETS gemm RUNTIME DESTINATION bin)
