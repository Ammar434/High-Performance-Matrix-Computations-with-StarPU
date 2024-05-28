#pragma once

/**
 * This file or its .cpp file should contain all kernel implementations that are called from StarPU
 * Note : Often kernels will need to call BLAS/LAPACK, an interface to these can  be found in blas.hpp and blas_extensions.hpp
 *        Any other functions will have to be implemented
 */

// Kernels in StarPU require a very specific interface
template <typename DataType>
void gemm_init_c(void* buffers[], void* cl_args);

template <typename DataType>
void gemm_1D_tile_cpu(void* buffers[], void* cl_args);

template <typename DataType>
void fill_value_matrix(void* buffers[], void* cl_args);

template <typename DataType>
void fill_value_matrix_cuda(void* buffers[], void* cl_args);

template <typename DataType>
void fill_value_matrix_random(void* buffers[], void* cl_args);

template <typename DataType>
void cuda_mult(void *buffers[], void *_args);

template <typename DataType>
void sum_matrix(void * buffers[], void* cl_args);

template <typename DataType>
void init_c(void* buffers[], void* cl_args);

template <typename DataType>
void assert_equal_cpu(void* buffers[], void* cl_args);

template <typename DataType>
void assert_equal_gpu(void* buffers[], void* cl_args);

template <typename DataType>
void print_cpu(void* buffers[], void* cl_args);

template <typename DataType>
void recursive_task_func(void *buffers[], void *cl_arg);

template <typename DataType>
int is_bubble(struct starpu_task *t , void *cl_arg);

template <typename DataType>
void gemm_gen_dag(struct starpu_task *t, void *cl_arg);

