#include "headers.hpp"

template <typename DataType>
__global__ void cuda_fill(DataType val, int m, int n, DataType *A, int ldA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n) {
    A[i + j * ldA] = val;
  }
}

template <typename DataType>
__global__ void cuda_test_equals(int m, int n, DataType *A, int ldA,
                                 DataType *B, int ldB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n) {
    if (A[i + j * ldA] != B[i + j * ldB]) {
      asm("trap;");
    }
  }
}

template <typename DataType>
void cuextensions<DataType>::fill(DataType val, int m, int n, DataType *A,
                                  int ldA, cudaStream_t stream) {
  if (m < 1) [[unlikely]] throw std::runtime_error("Invalid dimension m in cuextensions::fill");
  if (n < 1) [[unlikely]] throw std::runtime_error("Invalid dimension n in cuextensions::fill");
  if (ldA < m) [[unlikely]] throw std::runtime_error("Invalid dimension ldA in cuextensions::fill");
  
  dim3 dimBlock;
  dimBlock.x = 32;
  dimBlock.y = 16;
  dim3 dimGrid;
  dimGrid.x = static_cast<unsigned>(m - 1) / dimBlock.x + 1;
  dimGrid.y = static_cast<unsigned>(n - 1) / dimBlock.y + 1;

  cuda_fill<<<dimGrid, dimBlock, 0, stream>>>(val, m, n, A, ldA);
}

template <typename DataType>
bool cuextensions<DataType>::test_equals(int m, int n, DataType *A, int ldA,
                                         DataType *B, int ldB, cudaStream_t stream) {
  if (m < 1) [[unlikely]] throw std::runtime_error("Invalid dimension m in cuextensions::test_equals");
  if (n < 1) [[unlikely]] throw std::runtime_error("Invalid dimension n in cuextensions::test_equals");
  if (ldA < m) [[unlikely]] throw std::runtime_error("Invalid dimension ldA in cuextensions::test_equals");
  if (ldB < m) [[unlikely]] throw std::runtime_error("Invalid dimension ldB in cuextensions::test_equals");
    
  dim3 dimBlock;
  dimBlock.x = 32;
  dimBlock.y = 16;
  dim3 dimGrid;
  dimGrid.x = static_cast<unsigned>(m - 1) / dimBlock.x + 1;
  dimGrid.y = static_cast<unsigned>(n - 1) / dimBlock.y + 1;

  cuda_test_equals<<<dimGrid, dimBlock, 0, stream>>>(m, n, A, ldA, B, ldB);
  cudaStreamSynchronize(stream);
  return (cudaGetLastError() == cudaSuccess);
}

template struct cuextensions<float>;
template struct cuextensions<double>;
