#pragma once

/**
 * The matrix contains the tiles and operates on them
 * (TER PART 2) In the distributed case the distribution of tiles among nodes should be block-cyclic (https://netlib.org/scalapack/slug/node75.html)
 */
template <typename DataType>
struct Matrix
{
	// Size of the matrix
	uint32_t _height, _width;
	// Arangement of the tile
	uint32_t  _hTile, _wTile;
	// Size of the tile
	uint32_t _tileHeight, _tileWidth;
	// List of tiles
	//Tile<DataType> *_tiles;
	std::vector<Tile<DataType>> _tiles;
    // MPI tag allocate
    #ifdef USE_MPI
    int64_t _tags;
    #endif
	/*
	 * Constructor for a tile that is the size of the matrix
	 **/
	Matrix(/* TODO : Parameters*/
		   uint32_t width, uint32_t height);

	/*
	 * Constructor for the general matrix divided
	 * by the size of the tile
	 **/
	Matrix(uint32_t width, uint32_t height, uint32_t tileWidth, uint32_t tileHeight);
    // Destructor
    ~Matrix();
    /**
	 * This function can be used for debugging.
	 * It should print the contents of a tiled matrix
	 */
	void print();

	void fill_value(const DataType value);

	/*
	Fill the value with random float value
	*/
	void fill_random();

	/**
	 * This function can be used to check the correctness of operations.
	 * It should compare two tiled matrices and throw an exception if they are not equal
	 */
	void assert_equals(const Matrix &other);

	/**
	 * The gemm function does a generalised matrix multiplication on tiled matrices.
	 * It should compute C <- alpha * op(A) * op(B) + beta * C.
	 * alpha and beta are scalars
	 * A, B, and C are tiled matrices
	 * Each op is 'T' if the matrx is transposed, 'N' otherwise
	 */
	static void gemm(const DataType alpha, Matrix<DataType> &A, const char transA,
					 Matrix<DataType> &B, const char transB,
					 const DataType beta, Matrix<DataType> &C, bool redux, bool dag);
};
