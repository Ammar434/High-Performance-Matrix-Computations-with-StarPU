#pragma once

template <typename DataType>
struct blas {
	  static void gemm(
	  char transA,
	  char transB,
	  int m, int n, int k,
	  DataType alpha,
	  DataType* A, int ldA,
	  DataType* B, int ldB,
	  DataType beta,
	  DataType* C, int ldC);

	static void syrk(
		char uplo,
		char trans,
		int n, int k,
		DataType alpha,
		DataType* A, int ldA,
		DataType beta,
		DataType* C, int ldC);

	static void trsm(
		char side,
		char uplo,
		char trans,
		char diag,
		int m, int n,
		DataType alpha,
		DataType* A, int ldA,
		DataType* B, int ldB);
};

template <typename DataType>
struct lapack {
	static void potf2(
		char uplo,
		int n,
		DataType* A, int ldA,
		int& info);

	static void potrf(
		char uplo,
		int n,
		DataType* A, int ldA,
		int& info);
};

#ifdef USE_CUDA
template <typename DataType>
struct cublas {
	static void geam(
	  cublasHandle_t handle,
	  cublasOperation_t transA,
	  cublasOperation_t transB,
	  int m, int n,
	  const DataType alpha,
	  const DataType* A, int ldA,
	  const DataType beta,
	  const DataType* B, int ldB,
		DataType* C, int ldC);
	
	static void gemm(
	  cublasHandle_t handle,
	  cublasOperation_t transA,
	  cublasOperation_t transB,
	  int m, int n, int k,
	  const DataType alpha,
	  const DataType* A, int ldA,
	  const DataType* B, int ldB,
	  const DataType beta,
	  DataType* C, int ldC);

	static void syrk(
		cublasHandle_t handle,
		cublasFillMode_t uplo,
		cublasOperation_t trans,
		int n, int k,
		const DataType alpha,
		const DataType* A, int ldA,
		const DataType beta,
		DataType* C, int ldC);

	static void trsm(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t uplo,
		cublasOperation_t trans,
		cublasDiagType_t diag,
		int m, int n,
		const DataType alpha,
		const DataType* A, int ldA,
		DataType* B, int ldB);

	// static void potf2(
	// 	char uplo,
	// 	int n,
	// 	DataType* A, int ldA,
	// 	int& info);

	// static void potrf(
	// 	char uplo,
	// 	int n,
	// 	DataType* A, int ldA,
	// 	int& info);
};
#endif
