###
## This file contains a list of source files that must be compiled
## The first list is any normal cpp files
## The second list is source files with cuda code
###

set(cpp_sources
	src/blas.cpp
	src/kernels.cpp
	src/codelets.cpp
	src/tile.cpp
	src/matrix.cpp
)

set(cuda_sources
	src/blas_extensions.cu
	src/kernels.cu
)
