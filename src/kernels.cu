#include "headers.hpp"

#ifdef USE_CUDA
template void cuda_mult<float>(void *buffers[], void *_args);
template void cuda_mult<double>(void *buffers[], void *_args);
template void fill_value_matrix_cuda<float>(void *buffers[], void *_args);
template void fill_value_matrix_cuda<double>(void *buffers[], void *_args);

template void assert_equal_gpu<float>(void* buffers[], void* cl_args);
template void assert_equal_gpu<double>(void* buffers[], void* cl_args);

#endif
template <typename DataType>
void cuda_mult(void *buffers[], void *_args)
{
  DataType alpha, beta;
  DataType *subA;
  DataType *subB;
  DataType *subC;
  char transA, transB;
  int nxC, nyC, nyA;
  int ldA, ldB, ldC;
  starpu_codelet_unpack_args(_args, &alpha, &transA, &transB, &beta);
  beta = 1.0;
  subA = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
  subB = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[1]));
  subC = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[2]));

  nxC = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[2]));
  nyC = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[2]));
  nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));

  ldA = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[0]));
  ldB = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[1]));
  ldC = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[2]));
  //std::cout << "\n"
  //          << nxC << " " << nyC << " " << nyA << "\n";
  //cudaStream_t stream;
  cublasHandle_t handle;
  cublasCreate(&handle);
  //cudaStreamCreate(&stream);
  //cublasSetStream(handle,stream);
  cublas<DataType>::gemm(
      handle,
      (transA == 'N' || transA == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
      (transB == 'N' || transB == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
      nxC, nyC, nyA,
      alpha,
      subA, ldA,
      subB, ldB,
      beta,
      subC, ldC);
  cublasDestroy(handle);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

// static __global__ void cuda_mult_kernel(uint32_t nxC, uint32_t nyC, uint32_t nyA,
// 										uint32_t ldA, uint32_t ldB, uint32_t ldC,
// 										float *subA, float *subB, float *subC)
// {
// 	uint32_t id, i, j, k;
// 	float sum;
// 	id = blockIdx.x * blockDim.x + threadIdx.x;
// 	i = id % nxC;
// 	j = id / nxC;
// 	if (j >= nyC)
// 	{
// 		return;
// 	}
// 	sum = 0.;
// 	for (k = 0; k < nyA; k++)
// 	{
// 		sum += subA[i + k * ldA] * subB[k + j * ldB];
// 	}
// 	subC[i + j * ldC] = sum;
// }

// extern "C" void cuda_mult(void *descr[], void *arg)
// {
// 	(void)arg;
// 	float *d_subA, *d_subB, *d_subC;
// 	uint32_t nxC, nyC, nyA;
// 	uint32_t ldA, ldB, ldC;
// 	uint32_t nblocks;

// 	/* ptr gives a pointer to the first element of the local copy */
// 	d_subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
// 	d_subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
// 	d_subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);

// 	nxC = STARPU_MATRIX_GET_NX(descr[2]);
// 	nyC = STARPU_MATRIX_GET_NY(descr[2]);
// 	nyA = STARPU_MATRIX_GET_NY(descr[0]);

// 	ldA = STARPU_MATRIX_GET_LD(descr[0]);
// 	ldB = STARPU_MATRIX_GET_LD(descr[1]);
// 	ldC = STARPU_MATRIX_GET_LD(descr[2]);

// 	nblocks = (nxC * nyC + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
// 	cuda_mult<<<nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()>>>(nxC, nyC, nyA, ldA, ldB, ldC, d_subA, d_subB, d_subC);

// 	cudaError_t status = cudaGetLastError();
// 	if (status != cudaSuccess)
// 		STARPU_CUDA_REPORT_ERROR(status);
// }
template <typename DataType>
void fill_value_matrix_cuda(void* buffers[], void* cl_args)
{
  DataType* A;
  DataType value;
  int nxA, nyA, ldA;
  // unpack arguments
  starpu_codelet_unpack_args(cl_args, &value);

  // Matrix info
  A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
  nxA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
  nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));
  ldA = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[0]));
  // Cuda stream and Cublas  handle
  //cudaStream_t stream;
  cublasHandle_t handle;
  cublasCreate(&handle);
  //cudaStreamCreate(&stream);
  //cublasSetStream(handle,stream);
  // Fill matrix using cuda
  cuextensions<DataType>::fill(value, nxA, nyA, A,
                		  ldA, starpu_cuda_get_local_stream());

  // synchronize stream
  cudaStreamSynchronize(starpu_cuda_get_local_stream());

  // destroy resource
  cublasDestroy(handle);
  //cudaStreamDestroy(stream);

}


template <typename DataType>
void assert_equal_gpu(void* buffers[], void* cl_args)
{
    DataType* A, *B;
    int nxA, nyA, ldA, ldB;
    A = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[0]));
    nxA = static_cast<int>(STARPU_MATRIX_GET_NX(buffers[0]));
    nyA = static_cast<int>(STARPU_MATRIX_GET_NY(buffers[0]));
    ldA = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[0]));
    B = (DataType *)(STARPU_MATRIX_GET_PTR(buffers[1]));
    ldB = static_cast<int>(STARPU_MATRIX_GET_LD(buffers[1]));

    cublasHandle_t handle;
    cublasCreate(&handle);

    bool is_equal = cuextensions<DataType>::test_equals(nxA, nyA, A, ldA,
                                         B,ldB, starpu_cuda_get_local_stream());
    // synchronize stream
    cudaStreamSynchronize(starpu_cuda_get_local_stream());

    // destroy resource
    cublasDestroy(handle);
 
    if(is_equal)std::cout << "Equal\n";
    else std::cout << "Not Equal \n";
}
