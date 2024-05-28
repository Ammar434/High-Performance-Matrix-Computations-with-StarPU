#include "headers.hpp"

template void gemm_init_c<double>(void *buffers[], void *cl_args);
template void gemm_init_c<float>(void *buffers[], void *cl_args);

template void gemm_1D_tile_cpu<double>(void *buffers[], void *cl_args);
template void gemm_1D_tile_cpu<float>(void *buffers[], void *cl_args);

template void fill_value_matrix<float>(void *buffers[], void *_args);
template void fill_value_matrix<double>(void *buffers[], void *_args);

template void fill_value_matrix_random<float>(void *buffers[], void *_args);
template void fill_value_matrix_random<double>(void *buffers[], void *_args);

template void sum_matrix<double>(void *buffers[], void *cl_args);
template void sum_matrix<float>(void *buffers[], void *cl_args);

template void init_c<double>(void *buffers[], void *cl_args);
template void init_c<float>(void *buffers[], void *cl_args);

template void assert_equal_cpu<double>(void* buffers[], void* cl_args);
template void assert_equal_cpu<float>(void* buffers[], void* cl_args);

template void print_cpu<double>(void* buffers[], void* cl_args);
template void print_cpu<float>(void* buffers[], void* cl_args);

template int is_bubble<double>(struct starpu_task *t, void *cl_arg);
template int is_bubble<float>(struct starpu_task *t, void *cl_arg);

template void gemm_gen_dag<double>(struct starpu_task *t, void *cl_arg);
template void gemm_gen_dag<float>(struct starpu_task *t,  void *cl_arg);

template void recursive_task_func<double>(void *buffers[], void *cl_arg);
template void recursive_task_func<float>(void *buffers[], void *cl_arg);

template <typename DataType>
void gemm_init_c(void *buffers[], void *cl_args) {
    DataType* C;
    DataType beta;
    int nxc, nyc;
    C = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    // Get size
    nxc = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyc = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));
    starpu_codelet_unpack_args(cl_args, &beta);
    for(int i=0; i < nxc; ++i)
    {
        for(int j=0; j < nyc; ++j)
        {
            C[i * nxc + j] *= beta;
        }
    }

}
template <typename DataType>
void gemm_1D_tile_cpu(void *buffers[], void *cl_args)
{
    DataType alpha, beta;
    DataType *subA;
    DataType *subB;
    DataType *subC;
    char transA, transB;
    int nxC, nyC, nyA;
    int ldA, ldB, ldC;

    starpu_codelet_unpack_args(cl_args, &alpha, &transA, &transB, &beta);
    // Get Matrix
    beta = 1.0;
    subA = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    subB = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[1]));
    subC = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[2]));

    // Get size
    nxC = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[2]));
    nyC = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[2]));
    nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));

    // Get Ld
    ldA = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[0]));
    ldB = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[1]));
    ldC = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[2]));
    //std::cout << "\n" << nxC << " " << nyC << " " << nyA << "\n" ;
    // Do the blas gemm
    // We use OpenBLAS so I supposed if openmp is available it is used on the function
    blas<DataType>::gemm(transA, transB,
                         nxC, nyC, nyA,
                         alpha,
                         subA, ldA,
                         subB, ldB,
                         beta,
                         subC, ldC);
}

template <typename DataType>
void fill_value_matrix(void *buffers[], void *cl_args)
{
    DataType *A;
    int xA, yA;
    DataType value;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    xA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    yA = static_cast<int> (STARPU_MATRIX_GET_NY(buffers[0]));
    starpu_codelet_unpack_args(cl_args, &value);
    for (int i = 0; i < xA; ++i)
    {
        for (int j = 0; j < yA; ++j)
        {
            A[j + i * xA] = value;
        }
    } 
}

template <typename DataType>
void fill_value_matrix_random(void *buffers[], void *cl_args)
{
    DataType *A;
    int xA, yA;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    xA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    yA = static_cast<int> (STARPU_MATRIX_GET_NY(buffers[0]));
    for (int i = 0; i < xA; ++i)
    {
        for (int j = 0; j < yA; ++j)
        {
            A[j + i * xA] = static_cast<DataType>(starpu_drand48());
        }
    }
}

template <typename DataType>
void sum_matrix(void *buffers[], void *cl_args)
{
    DataType *A;
    DataType *B;
    int nxA, nyA;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    B = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[1]));

    // Get size
    nxA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));

    // Get Ld
    for(int i=0; i < nxA; ++i)
    {
        for(int j=0; j < nyA; ++j)
        {
            A[i * nxA + j] = A[i * nxA + j] + B[i * nxA +j];
        }
    }
}

template <typename DataType>
void init_c(void *buffers[], void *cl_args) {
    DataType* C;
    int nxc, nyc;
    C = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    // Get size
    nxc = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyc = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));
    
    for(int i=0; i < nxc; ++i)
    {
        for(int j=0; j < nyc; ++j)
        {
            C[i * nxc + j] = static_cast<DataType>( 0.0);
        }
    }

}


template <typename DataType>
void assert_equal_cpu(void* buffers[], void* cl_args)
{
    DataType *A;
    DataType *B;
    int nxA, nyA;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    B = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[1]));

    // Get size
    nxA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));
    bool is_equal = true;
    // Get Ld
    for(int i=0; i < nxA; ++i)
    {
        for(int j=0; j < nyA; ++j)
        {
            if(A[i * nxA + j] != B[i + nxA +j]){
                if(is_equal)std::cout << "Not equal \n";
                is_equal = false;
            }
        }
    }
    if(is_equal)std::cout << "Equal\n";
}

template <typename DataType>
void print_cpu(void* buffers[], void* cl_args)
{
    DataType *A;
    int nxA, nyA;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));

    // Get size
    nxA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));

    // Get Ld
    for(int i=0; i < nxA; ++i)
    {
        for(int j=0; j < nyA; ++j)
        {
            std::cout << std::setprecision(4) << A[i * nxA + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename DataType>
void recursive_task_func(void *buffers[], void *cl_arg)
{
    assert(0);
    return;
}

template <typename DataType>
int is_bubble(struct starpu_task *t, void *cl_arg)
{
    (void)t;
    (void) cl_arg;
    return 1;
}

template <typename DataType>
void gemm_gen_dag(struct starpu_task *t, void *cl_arg)
{
    DataType alpha, beta;
    char transA, transB;
    int i;
    struct gemm_bubble_task_arg<DataType> *value = (struct gemm_bubble_task_arg<DataType>*)  cl_arg;
    starpu_data_handle_t subA = (starpu_data_handle_t) value->A; 
    starpu_data_handle_t subB = (starpu_data_handle_t) value->B; 
    starpu_data_handle_t subC = (starpu_data_handle_t) value->C; 
    alpha = static_cast<DataType>(value->alpha);
    beta = 1.0;
    transA = static_cast<char>(value->transA);
    transB = static_cast<char>(value->transB);
    for(i=0; i < PARTS; ++i)
    {
        int indexA = (i/2) * 2;
        int indexB = (i%2);
        #ifndef USE_MPI
        starpu_task_insert(&gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, starpu_data_get_sub_data(subA, 1, indexA),
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, starpu_data_get_sub_data(subB, 1, indexB),
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, starpu_data_get_sub_data(subC, 1, i),
                       0);
         starpu_task_insert(&gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, starpu_data_get_sub_data(subA, 1, indexA+1),
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, starpu_data_get_sub_data(subB, 1, indexB+2),
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, starpu_data_get_sub_data(subC, 1, i),
                       0);
                
        #else
         starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, starpu_data_get_sub_data(subA, 1, indexA),
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, starpu_data_get_sub_data(subB, 1, indexB),
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, starpu_data_get_sub_data(subC, 1, i),
                       0);
         starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, starpu_data_get_sub_data(subA, 1, indexA+1),
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, starpu_data_get_sub_data(subB, 1, indexB+2),
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, starpu_data_get_sub_data(subC, 1, i),
                       0);
        #endif

    }
   free(value); 
}
