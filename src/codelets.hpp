#pragma once
#include "headers.hpp"

/**
 * This file or its .cpp file should contain the implementation of all codelets needed.
 */

static struct starpu_perfmodel model = {
    .type=STARPU_HISTORY_BASED,
    .symbol="gemm_cl",
};

template <typename DataType>
struct starpu_codelet get_gemm_cl() {
    return {
        .cpu_funcs = {gemm_1D_tile_cpu<DataType>},
#ifdef USE_CUDA
        /* CUDA implementation of the codelet */
        // .where = STARPU_CUDA,
        .cuda_funcs = {cuda_mult<DataType>},
        .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
        /* the codelet manipulates 3 buffers that are managed by the DSM */
        .nbuffers = 3,
	.model = &model,
        //.modes = {STARPU_R, STARPU_R,static_cast<starpu_data_access_mode> (STARPU_RW|STARPU_COMMUTE)},
        //.modes = {STARPU_R, STARPU_R,STARPU_REDUX},
        .name = "gemm",
	};
}

template <typename DataType>
struct starpu_codelet get_fill_cl() {
    return {
        .cpu_funcs = {fill_value_matrix<DataType>},
//#ifdef STARPU_USE_CUDA
        /* CUDA implementation of the codelet */
        // .where = STARPU_CUDA,
//        .cuda_funcs = {fill_value_matrix_cuda<DataType>},
//        .cuda_flags = {STARPU_CUDA_ASYNC},
//#endif
        /* the codelet manipulates 1 buffers that are managed by the DSM */
        .nbuffers = 1,
        .modes = {STARPU_W},
        .name = "fill_matrix",
	};
}

template <typename DataType>
struct starpu_codelet get_fill_random_cl() {
    return {
        .cpu_funcs = {fill_value_matrix_random<DataType>},
       /* the codelet manipulates 1 buffers that are managed by the DSM */
        .nbuffers = 1,
        .modes = {STARPU_W},
        .name = "fill_matrix",
	};
}
template <typename DataType>
struct starpu_codelet get_init_c(){
    return {
        .cpu_funcs = {init_c<DataType>},
        .nbuffers = 1,
        .modes = {STARPU_W},
        .name = "init_c",
    };
}

template <typename DataType>
static starpu_codelet init_cl = get_init_c<DataType>();

template <typename DataType>
struct starpu_codelet get_sum_matrix_cl()
{
    return{
        .cpu_funcs = {sum_matrix<DataType>},
        .nbuffers = 2,
        .modes = {static_cast<starpu_data_access_mode>(STARPU_RW|STARPU_COMMUTE), STARPU_R},
        .name = "sum_matrix",
    };
}

template <typename DataType>
static starpu_codelet sum_matrix_cl = get_sum_matrix_cl<DataType>();


template<typename DataType>
struct starpu_codelet get_assert_equal_cl()
{
    return{
        .cpu_funcs = {assert_equal_cpu<DataType>},
        #ifdef USE_CUDA
        //.cuda_funcs = {assert_equal_gpu<DataType>},
        #endif
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_R},
        .name = "is_equal",
    };
}

template <typename DataType>
struct starpu_codelet get_print_cl()
{
    return{
        .cpu_funcs = {print_cpu<DataType>},
        .nbuffers = 1,
        .modes = {STARPU_R},
        .name = "print",
    };
}

template <typename DataType>
static starpu_codelet gemm_cl = get_gemm_cl<DataType>();

template <typename DataType>
static starpu_codelet fill_cl = get_fill_cl<DataType>();

template <typename DataType>
static starpu_codelet fill_random_cl = get_fill_random_cl<DataType>();

template <typename DataType>
static starpu_codelet assert_equal_cl = get_assert_equal_cl<DataType>();

template <typename DataType>
static starpu_codelet print_cl = get_print_cl<DataType>();

template<typename DataType>
struct gemm_bubble_task_arg  
{
    DataType alpha;
    char transA;
    starpu_data_handle_t A;
    char transB;
    starpu_data_handle_t B;
    DataType beta; 
    starpu_data_handle_t C;
};

static starpu_data_filter block_filter_for_matrix = 
{    
    .filter_func = starpu_matrix_filter_block,
    .nchildren = PARTS,
}; 

template <typename DataType>
struct starpu_codelet get_recursive_gemm()
{
    return {
        .cpu_funcs = {recursive_task_func<DataType>},
        .bubble_func = is_bubble<DataType>,
        .bubble_gen_dag_func= gemm_gen_dag<DataType>,
        .nbuffers = 0,
        //.modes = {STARPU_R,STARPU_R,static_cast<starpu_data_access_mode> (STARPU_RW|STARPU_COMMUTE)},
        .name = "recursive_block",
    };
}

template<typename DataType>
static starpu_codelet recursive_gemm = get_recursive_gemm<DataType>();

template<typename DataType>
struct starpu_codelet get_gemm_init_c()
{
    return {
        .cpu_funcs= {gemm_init_c<DataType>},
        .nbuffers = 1,
        //.modes = {STARPU_RW},
        .modes = {static_cast<starpu_data_access_mode> (STARPU_RW|STARPU_COMMUTE)},
        .name = "gemm_init_c",
    };
}

template<typename DataType>
static starpu_codelet gemm_init = get_gemm_init_c<DataType>();
