#pragma once

/**
 * The tile class is an abstraction to hold StarPU matrix interfaces and launch tasks on them
 */
template <typename DataType>
struct Tile
{
	uint32_t _height, _width;
	DataType *_tile;
	starpu_data_handle_t _handle;
	// Constructor
	Tile(uint32_t width, uint32_t height);
	Tile(DataType *matrix, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY);
	Tile(uint32_t width, uint32_t height, int rank, bool belonging,uint32_t belong , uint32_t tag);
    // Move Constructor
    Tile(Tile&& tile);
    // Destructor
    ~Tile();
	void fill_value(const DataType value);

	void fill_random();

	void print();

	void assert_equals(const Tile &other);

    static void init_gemm(const DataType beta,Tile<DataType> &C);

	static void gemm(const DataType alpha, Tile<DataType> &A, const char transA,
					 Tile<DataType> &B, const char transB,
					 const DataType beta, Tile<DataType> &C, bool redux, bool dag);
};
