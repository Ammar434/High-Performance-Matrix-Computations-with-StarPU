#pragma once
#ifdef STARPU_USE_CUDA 

template <typename DataType> struct cuextensions {
  static void fill(DataType val, int m, int n, DataType *A, int ldA,
                   cudaStream_t stream = 0);
  static bool test_equals(int m, int n, DataType *A, int ldA, DataType *B,
                          int ldB, cudaStream_t stream = 0);
};

#endif