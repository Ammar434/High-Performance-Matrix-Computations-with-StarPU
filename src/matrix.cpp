#include "headers.hpp"

template struct Matrix<double>;
template struct Matrix<float>;

template <typename DataType>
Matrix<DataType>::Matrix(uint32_t width, uint32_t height)
{
    _width = width;
    _height = height;
    // Declare number of Tile
    _hTile = 1;
    _wTile = 1;
    // Declare size of the tile
    _tileWidth = width;
    _tileHeight = height;

    // Alloc using starpu
    starpu_malloc((void **)&_tiles, sizeof(Tile<DataType>));
    _tiles.push_back(Tile<DataType>(_tileWidth, _tileHeight));
}

template <typename DataType>
Matrix<DataType>::Matrix(uint32_t width, uint32_t height, uint32_t tileWidth, uint32_t tileHeight)
{
    // Declare size of the matrix
    _width = width;
    _height = height;
    // Declare number of Tile
    _hTile = height / tileHeight;
    _wTile = width / tileWidth;
    // Declare size of the tile
    _tileWidth = tileWidth;
    _tileHeight = tileHeight;
    std::cout << "width: "<< _tileWidth << "; height: " << _tileHeight << "\n";
    // Allocation of the tile
    #ifdef USE_MPI
    int rank , size;
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    _tags = starpu_mpi_tags_allocate(_wTile*_hTile+1);
    #endif

    for(uint32_t i = 0; i < _wTile; ++i )
    {
        for(uint32_t j = 0; j < _hTile; ++j)
        {
            #ifdef USE_MPI
            _tiles.push_back(
                    Tile<DataType>(_tileWidth, _tileHeight,
                        rank, (uint32_t)rank == ((i * _hTile + j)%(uint32_t)size)
                        ,(i * _wTile + j)%(uint32_t)size
                        ,(uint32_t)_tags + i * _hTile + j )
                    );
	        #else
            _tiles.push_back(Tile<DataType>(_tileWidth, _tileHeight));
            #endif
        }
    }
    starpu_task_wait_for_all(); 
    
}

template <typename DataType>
Matrix<DataType>::~Matrix()
{
    //for(auto tile: _tiles)
    //{
    //   starpu_data_unregister(tile._handle);
    //   free( tile._tile); 
    //}
    //std::cout << "delete tag \n";
#ifdef USE_MPI
    starpu_mpi_tags_free(_tags);
#endif
}


template <typename DataType>
void Matrix<DataType>::print()
{
    // TODO Faire un print correctement
    std::cout << _wTile << " " << _hTile << "\n";
    for (uint32_t i = 0; i < _wTile * _hTile; ++i)
    {
	    std::cout << "\n" << i <<" " << _tileWidth << " " << _tileHeight << "\n";
	    _tiles[i].print();
	    std::cout << "\n";
        starpu_task_wait_for_all(); 
    }
    starpu_task_wait_for_all(); 

}

template <typename DataType>
void Matrix<DataType>::fill_value(const DataType value)
{
    for (uint32_t i = 0; i < _wTile *_hTile; ++i)
    {
	    _tiles[i].fill_value(value);
    }
        starpu_task_wait_for_all();
}

template <typename DataType>
void Matrix<DataType>::fill_random()
{

    for (uint32_t i = 0; i < _wTile; i++)
    {
        for (uint32_t j = 0; j < _hTile; j++)
        {
		    _tiles[i + j * _wTile].fill_random();
       	}
    }
    starpu_task_wait_for_all();
}

template <typename DataType>
void Matrix<DataType>::assert_equals(const Matrix &other)
{
    if (_height != other._height || _width != other._width)
        throw std::invalid_argument("The two matrices are not of the same size");
    for (uint32_t i = 0; i < _wTile; ++i)
    {
        for (uint32_t j = 0; j < _hTile; ++j)
        {
            _tiles[i + j * _wTile].assert_equals(other._tiles[i + j * _wTile]);
        }
    }
    starpu_task_wait_for_all(); 
}

template <typename DataType>
void Matrix<DataType>::gemm(const DataType alpha, Matrix<DataType> &A, const char transA,
                            Matrix<DataType> &B, const char transB,
                            const DataType beta, Matrix<DataType> &C, bool redux, bool dag)
{
    if (A._width != B._height)
    {
	std::cout << A._width << " A !=B " << B._height << "\n";
        throw std::invalid_argument("A and B dim doesnt match up matrix ");
    }
    uint32_t ldA, ldB;
    if(transA == 'N')ldA = A._wTile;
    else ldA = A._hTile;
    if(transB == 'N')ldB = B._wTile;
    else ldB = B._hTile;
    // First step of gemm
    for(uint32_t i = 0; i < C._hTile * C._wTile; ++i)
    {
        Tile<DataType>::init_gemm(beta, C._tiles[i]);
    }
    starpu_task_wait_for_all();
    // Partition matrix for DAG
    if(!dag && redux)
    {
        for(uint32_t i=0; i< C._hTile*C._wTile;++i)
        {
            starpu_data_set_reduction_methods(C._tiles[i]._handle, &sum_matrix_cl<DataType>, &init_cl<DataType>);
        }
    }
    else if(dag)
    {
        for(uint32_t i=0; i< C._hTile*C._wTile;++i)
        {
            starpu_data_partition(C._tiles[i]._handle, &block_filter_for_matrix);
        }
        for(uint32_t i=0; i< A._hTile*A._wTile;++i)
        {
            starpu_data_partition(A._tiles[i]._handle, &block_filter_for_matrix);
            starpu_data_partition(B._tiles[i]._handle, &block_filter_for_matrix);
        }
    }
    std::cout << "Je passe les partition\n";
    // Compute all gemm tiling
    for(uint32_t i = 0; i < C._hTile; ++i)
    {
	    for(uint32_t j = 0; j < C._wTile; ++j)
	    {
	        for(uint32_t k = 0; k < A._hTile; ++k)
            {
	         Tile<DataType>::gemm(alpha, A._tiles[j + k * ldA], transA,
				     B._tiles[k + i *  ldB], transB,
				     beta, C._tiles[j + i * C._wTile], redux, dag );
	        }
        }
    }	
    starpu_task_wait_for_all();
    // Departition All if DAG
    if(dag)
    {
        for(uint32_t i=0; i< C._hTile*C._wTile;++i)
        {
            starpu_data_unpartition(C._tiles[i]._handle, STARPU_MAIN_RAM);
        }
        for(uint32_t i=0; i< A._hTile*A._wTile;++i)
        {
            starpu_data_unpartition(A._tiles[i]._handle, STARPU_MAIN_RAM);
            starpu_data_unpartition(B._tiles[i]._handle, STARPU_MAIN_RAM);
        }
    }

}
