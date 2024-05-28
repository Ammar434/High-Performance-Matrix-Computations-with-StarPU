#include "headers.hpp"
#include "blas_extern.hpp"

template <typename DataType>
void blas<DataType>::gemm(
	char transA,
	char transB,
	int m, int n, int k,
	DataType alpha,
	DataType* A, int ldA,
	DataType* B, int ldB,
	DataType beta,
	DataType* C, int ldC
) {
	if constexpr (std::is_same_v<DataType, float>) {
		sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC);
	}
}

template <typename DataType>
void blas<DataType>::syrk(
	char uplo,
	char trans,
	int n, int k,
	DataType alpha,
	DataType* A, int ldA,
	DataType beta,
	DataType* C, int ldC
) {
	if constexpr (std::is_same_v<DataType, float>) {
		ssyrk_(&uplo, &trans, &n, &k, &alpha, A, &ldA, &beta, C, &ldC);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dsyrk_(&uplo, &trans, &n, &k, &alpha, A, &ldA, &beta, C, &ldC);
	}
}

template <typename DataType>
void blas<DataType>::trsm(
	char side,
	char uplo,
	char trans,
	char diag,
	int m, int n,
	DataType alpha,
	DataType* A, int ldA,
	DataType* B, int ldB
) {
	if constexpr (std::is_same_v<DataType, float>) {
		strsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &ldA, B, &ldB);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dtrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &ldA, B, &ldB);
	}
}

template struct blas<float>;
template struct blas<double>;

template <typename DataType>
void lapack<DataType>::potf2(
	char uplo,
	int n,
	DataType* A, int ldA,
	int& info
) {
	if constexpr (std::is_same_v<DataType, float>) {
		spotf2_(&uplo, &n, A, &ldA, &info);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dpotf2_(&uplo, &n, A, &ldA, &info);
	}
}

template <typename DataType>
void lapack<DataType>::potrf(
	char uplo,
	int n,
	DataType* A, int ldA,
	int& info
) {
	if constexpr (std::is_same_v<DataType, float>) {
		spotrf_(&uplo, &n, A, &ldA, &info);
	}

	if constexpr (std::is_same_v<DataType, double>) {
		dpotrf_(&uplo, &n, A, &ldA, &info);
	}
}

#ifdef USE_CUDA
template <typename DataType>
void cublas<DataType>::gemm(
	cublasHandle_t handle,
	cublasOperation_t transA,
	cublasOperation_t transB,
	int m, int n, int k,
	const DataType alpha,
	const DataType* A, int ldA,
	const DataType* B, int ldB,
	const DataType beta,
	DataType* C, int ldC
) {
	cublasStatus_t stat;

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgemm(handle, transA, transB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC);
	}
	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDgemm(handle, transA, transB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Error in cuBLAS gemm" << std::endl;
		std::cerr << cublasGetStatusName(stat) << std::endl;
		std::cerr << cublasGetStatusString(stat) << std::endl;
	}
}

template <typename DataType>
void cublas<DataType>::geam(
	cublasHandle_t handle,
	cublasOperation_t transA,
	cublasOperation_t transB,
	int m, int n,
	const DataType alpha,
	const DataType* A, int ldA,
	const DataType beta,
	const DataType* B, int ldB,
	DataType* C, int ldC
) {
	cublasStatus_t stat;

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSgeam(handle, transA, transB, m, n, &alpha, A, ldA, &beta, B, ldB, C, ldC);
	}
	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDgeam(handle, transA, transB, m, n, &alpha, A, ldA, &beta, B, ldB, C, ldC);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Error in cuBLAS geam" << std::endl;
		std::cerr << cublasGetStatusName(stat) << std::endl;
		std::cerr << cublasGetStatusString(stat) << std::endl;
	}
}

template <typename DataType>
void cublas<DataType>::syrk(
	cublasHandle_t handle,
	cublasFillMode_t uplo,
	cublasOperation_t trans,
	int n, int k,
	const DataType alpha,
	const DataType* A, int ldA,
	const DataType beta,
	DataType* C, int ldC
) {
	cublasStatus_t stat;

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasSsyrk(handle, uplo, trans, n, k, &alpha, A, ldA, &beta, C, ldC);
	}
	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDsyrk(handle, uplo, trans, n, k, &alpha, A, ldA, &beta, C, ldC);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Error in cuBLAS syrk" << std::endl;
		std::cerr << cublasGetStatusName(stat) << std::endl;
		std::cerr << cublasGetStatusString(stat) << std::endl;
	}
}

template <typename DataType>
void cublas<DataType>::trsm(
	cublasHandle_t handle,
	cublasSideMode_t side,
	cublasFillMode_t uplo,
	cublasOperation_t trans,
	cublasDiagType_t diag,
	int m, int n,
	const DataType alpha,
	const DataType* A, int ldA,
	DataType* B, int ldB
) {
	cublasStatus_t stat;

	if constexpr (std::is_same_v<DataType, float>) {
		stat = cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, ldA, B, ldB);
	}
	if constexpr (std::is_same_v<DataType, double>) {
		stat = cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, ldA, B, ldB);
	}

	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Error in cuBLAS trsm" << std::endl;
		std::cerr << cublasGetStatusName(stat) << std::endl;
		std::cerr << cublasGetStatusString(stat) << std::endl;
	}
}

template struct cublas<float>;
template struct cublas<double>;
#endif
