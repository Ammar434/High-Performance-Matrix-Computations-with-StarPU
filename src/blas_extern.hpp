extern "C" void sgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	float* alpha,
	float* A,
	int* lda,
	float* B,
	int* ldb,
	float* beta,
	float* C,
	int* ldc
);

extern "C" void dgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	double* alpha,
	double* A,
	int* lda,
	double* B,
	int* ldb,
	double* beta,
	double* C,
	int* ldc
);

extern "C" void ssyrk_(
	char* uplo,
	char* trans,
	int* n,
	int* k,
	float* alpha,
	float* A,
	int* lda,
	float* beta,
	float* C,
	int* ldc
);

extern "C" void dsyrk_(
	char* uplo,
	char* trans,
	int* n,
	int* k,
	double* alpha,
	double* A,
	int* lda,
	double* beta,
	double* C,
	int* ldc
);

extern "C" void strsm_(
	char* side,
	char* uplo,
	char* trans,
	char* diag,
	int* m,
	int* n,
	float* alpha,
	float* A,
	int* lda,
	float* B,
	int* ldb
);

extern "C" void dtrsm_(
	char* side,
	char* uplo,
	char* trans,
	char* diag,
	int* m,
	int* n,
	double* alpha,
	double* A,
	int* lda,
	double* B,
	int* ldb
);

extern "C" void spotf2_(
	char *uplo,
	int *n,
	float *A,
	int *ldA,
	int *info
);

extern "C" void dpotf2_(
	char *uplo,
	int *n,
	double *A,
	int *ldA,
	int *info
);

extern "C" void spotrf_(
	char *uplo,
	int *n,
	float *A,
	int *ldA,
	int *info
);

extern "C" void dpotrf_(
	char *uplo,
	int *n,
	double *A,
	int *ldA,
	int *info
);
