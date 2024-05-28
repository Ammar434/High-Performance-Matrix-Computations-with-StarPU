/**
 * This file contains header files that are used by the code, if you create any
 * more .hpp files do not forget to include them here. Any .cpp files will
 * require you to update the sources.cmake with it's filename.
 */

#define PARTS 4
#include <cassert>
#include <chrono>
#include <cstddef>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <bit>
#include <iomanip>
#include <unistd.h>
#include <omp.h>
#include <starpu.h>
extern "C" { 
    #include <starpu_disk.h>
}
#ifdef USE_MPI
#include <mpi.h>
#include <starpu_mpi.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <starpu_cublas_v2.h>
#endif

#include "arg_parser.hpp"
#include "blas.hpp"
#include "blas_extensions.hpp"
#include "kernels.hpp"
#include "codelets.hpp"
#include "tile.hpp"
#include "matrix.hpp"
