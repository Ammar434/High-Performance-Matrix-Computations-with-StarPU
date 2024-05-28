#include "headers.hpp"

template <typename DataType>
Tile<DataType>::Tile(uint32_t width, uint32_t height)
{
    // Put size of the matrix
    _width = width;
    _height = height;

    // Using starpu malloc we alloc the matrix
    starpu_malloc((void **)&_tile, _width * _height * sizeof(DataType));
    //starpu_matrix_data_register(&_handle, STARPU_MAIN_RAM, (uintptr_t)_tile, _width,_width, _height, sizeof(DataType));
    // Used when out of core support
    starpu_matrix_data_register(&_handle, -1, (uintptr_t)nullptr, _width,_width, _height, sizeof(DataType));
}

template <typename DataType>
Tile<DataType>::Tile(DataType *matrix, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY)
{
    _width = width;
    _height = height;
    starpu_malloc((void **)&_tile, _width * _height * sizeof(DataType));

    for (uint32_t i = 0; i < _height; ++i)
    {
        for (uint32_t j = 0; j < _width; ++j)
        {
            //_tile[i + j * _height] = matrix[ _width  *  posX  + i + _height * (posY+j)];
            _tile[i + j * _height] = matrix[posX + i * _width + posY + j];
            std::cout << (posY + j) * _width + (posX + i) << " ";
        }
        std::cout << "\n";
    }
}
template <typename DataType>
Tile<DataType>::Tile(uint32_t width, uint32_t height, int rank, bool belonging, uint32_t belong  ,uint32_t tag)
{
    // Put size of the matrix
    _width = width;
    _height = height;

    // Using starpu malloc we alloc the matrix
    #ifdef USE_MPI
    starpu_malloc((void **)&_tile, _width * _height * sizeof(DataType));
    if(!belonging)
    {
        starpu_matrix_data_register(&_handle, -1, (uintptr_t)nullptr, _width,_width, _height, sizeof(DataType));
    }
    else
    {
        //starpu_matrix_data_register(&_handle, STARPU_MAIN_RAM, (uintptr_t)_tile, _width,_width, _height, sizeof(DataType));
        // Used when out of core support is here 
        starpu_matrix_data_register(&_handle, -1, (uintptr_t)nullptr, _width,_width, _height, sizeof(DataType));
    }
    starpu_mpi_data_register(_handle, tag,(int)belong);
    #endif
}


template <typename DataType>
Tile<DataType>::Tile(Tile<DataType>&& tile)
{
   //std::cout<< "I delete myself\n";
  _width = tile._width;
  _height = tile._height;
  //_tile = std::move(tile._tile);
  _handle = std::move(tile._handle);
   tile._handle = nullptr;
  //tile._tile = nullptr;
}

template <typename DataType>
Tile<DataType>::~Tile()
{
    if(_handle == nullptr)return;
    starpu_data_unregister(_handle);
    //if(_tile == nullptr)return;
    //delete  _tile;
}

template <typename DataType>
void Tile<DataType>::print()
{
    #ifndef USE_MPI
    starpu_task_insert(&print_cl<DataType>,
                       STARPU_R, _handle,
                       0);
    #else
    starpu_mpi_task_insert(MPI_COMM_WORLD ,&print_cl<DataType>,
                       STARPU_R, _handle,
                       0);
    #endif

}

template <typename DataType>
void Tile<DataType>::fill_value(DataType value)
{
    #ifndef USE_MPI
    starpu_task_insert(&fill_cl<DataType>,
                       STARPU_VALUE, &value, sizeof(value),
                       STARPU_W, _handle,
                       0);
    #else
    starpu_mpi_task_insert(MPI_COMM_WORLD ,&fill_cl<DataType>,
                       STARPU_VALUE, &value, sizeof(value),
                       STARPU_W, _handle,
                       0);
    #endif

}

template <typename DataType>
void Tile<DataType>::fill_random()
{
    starpu_srand48((int)time(NULL));
    #ifndef USE_MPI
    starpu_task_insert(&fill_random_cl<DataType>,
                       STARPU_W, _handle,
                       0);
    #else
    starpu_mpi_task_insert(MPI_COMM_WORLD ,&fill_random_cl<DataType>,
                       STARPU_W, _handle,
                       0);
    #endif
    
}

template <typename DataType>
void Tile<DataType>::assert_equals(const Tile &other)
{
    /*for (uint32_t i = 0; i < _height; ++i)
    {
        for (uint32_t j = 0; j < _width; ++j)
        {
            if (_tile[i * _width + j] != other._tile[i * _width + j])
            {
                std::cout << "Not equal\n";
                return;
            }
        }
    }
    std::cout << "Equal\n";*/
    #ifndef USE_MPI
    starpu_task_insert(&assert_equal_cl<DataType>,
                       STARPU_R, _handle,
                       STARPU_R, other._handle,
                       0);
    #else
    starpu_mpi_task_insert(MPI_COMM_WORLD ,&assert_equal_cl<DataType>,
                       STARPU_R, _handle,
                       STARPU_R, other._handle,
                       0);
    #endif

}

template <typename DataType>
void Tile<DataType>::init_gemm(const DataType beta, Tile<DataType> &C)
{
#ifndef USE_MPI
    starpu_task_insert(&gemm_init<DataType>,
                    STARPU_VALUE, &beta, sizeof(beta),
                    STARPU_RW|STARPU_COMMUTE,C._handle,
                    0);
#else
    starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_init<DataType>,
                    STARPU_VALUE,&beta, sizeof(beta),
                    STARPU_RW|STARPU_COMMUTE,C._handle,
                    0);
#endif
}

template <typename DataType>
void Tile<DataType>::gemm(const DataType alpha, Tile<DataType> &A, const char transA,
                          Tile<DataType> &B, const char transB,
                          const DataType beta, Tile<DataType> &C, bool redux, bool dag)
{
    //std::cout << A._width << " " << A._height << "\n";
    //std::cout << B._width << " " << B._height << "\n";
    // Test if it is possible to make gemm multiplication
    if (A._width != B._height)
    {
        std::cout << A._width << " A !=B " << B._height << "\n";
        throw std::invalid_argument("A and B dim doesnt match up");
    }
    // Gemm task
    #ifndef USE_MPI
        if(!redux && !dag)
        {
        starpu_task_insert(&gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, A._handle,
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, B._handle,
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, C._handle,
#ifdef STARPU_OPENMP
                       STARPU_POSSIBLY_PARALLEL, 1,
#endif
                       0);
        }
        else if(dag)
        {
            gemm_bubble_task_arg<DataType> *dag= nullptr; 
            dag = new gemm_bubble_task_arg<DataType>{};
            dag->A = A._handle;
            dag->B = B._handle;
            dag->C = C._handle;
            dag->alpha = alpha;
            dag->beta = beta;
            dag->transA = transA;
            dag->transB = transB;
            starpu_task_insert(&recursive_gemm<DataType>,
                       //STARPU_VALUE, &alpha, sizeof(alpha),
                       //STARPU_VALUE, &transA, sizeof(transA),
                       //STARPU_R, A._handle,
                       //STARPU_VALUE, &transB, sizeof(transB),
                       //STARPU_R, B._handle,
                       //STARPU_VALUE, &beta, sizeof(beta),
                       //STARPU_RW|STARPU_COMMUTE, C._handle,
                       STARPU_BUBBLE_GEN_DAG_FUNC_ARG, dag,
                       0);

        }
        else
        {
           starpu_task_insert(&gemm_cl<DataType>,
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_VALUE, &transA, sizeof(transA),
                STARPU_R, A._handle,
                STARPU_VALUE, &transB, sizeof(transB),
                STARPU_R, B._handle,
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_REDUX, C._handle,
                0);
        }
    #else
        if(!redux)
        {
        starpu_mpi_task_insert(MPI_COMM_WORLD, &gemm_cl<DataType>,
                       STARPU_VALUE, &alpha, sizeof(alpha),
                       STARPU_VALUE, &transA, sizeof(transA),
                       STARPU_R, A._handle,
                       STARPU_VALUE, &transB, sizeof(transB),
                       STARPU_R, B._handle,
                       STARPU_VALUE, &beta, sizeof(beta),
                       STARPU_RW|STARPU_COMMUTE, C._handle,
#ifdef STARPU_OPENMP
                       STARPU_POSSIBLY_PARALLEL, 1,
#endif

                       0);
        }
        else if(dag)
        {
            gemm_bubble_task_arg<DataType> *dag= nullptr; 
            dag = new gemm_bubble_task_arg<DataType>{};
            dag->A = A._handle;
            dag->B = B._handle;
            dag->C = C._handle;
            dag->alpha = alpha;
            dag->beta = beta;
            dag->transA = transA;
            dag->transB = transB;
            starpu_mpi_task_insert(MPI_COMM_WORLD,&recursive_gemm<DataType>,
                       //STARPU_VALUE, &alpha, sizeof(alpha),
                       //STARPU_VALUE, &transA, sizeof(transA),
                       //STARPU_R, A._handle,
                       //STARPU_VALUE, &transB, sizeof(transB),
                       //STARPU_R, B._handle,
                       //STARPU_VALUE, &beta, sizeof(beta),
                       //STARPU_RW|STARPU_COMMUTE, C._handle,
                       STARPU_BUBBLE_GEN_DAG_FUNC_ARG, dag,
                       0);


        }
        else
        {
        //starpu_data_set_reduction_methods(C._handle, &sum_matrix_cl<DataType>, &init_cl<DataType>);
        starpu_mpi_task_insert(MPI_COMM_WORLD ,&gemm_cl<DataType>,
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_VALUE, &transA, sizeof(transA),
                STARPU_R, A._handle,
                STARPU_VALUE, &transB, sizeof(transB),
                STARPU_R, B._handle,
                STARPU_VALUE, &beta, sizeof(beta),
                STARPU_REDUX, C._handle,
                0);
        starpu_mpi_redux_data(MPI_COMM_WORLD, C._handle);
        }
    #endif
}

template struct Tile<double>;
template struct Tile<float>;
