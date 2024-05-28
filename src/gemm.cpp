#include "headers.hpp"
#define SIZE 64
#define TILENUMBER 2


void printHelp()
{
  std::cout << "Parameters for gemm:\n"
               "  --help            --  Print this help\n"
               "  --type [d/s]      --  Set the data type (double/single)\n"
               "  --m [uint]        --  Set the size of M\n"
               "  --n [uint]        --  Set the size of N\n"
               "  --k [uint]        --  Set the size of K\n"
               "  --bs [uint]       --  Set the block size\n"
               "  --transA [N/T]    --  Set transposition value for A\n"
               "  --transB [N/T]    --  Set transposition value for B\n"
               "  --alpha [float]   --  Set value for alpha\n"
               "  --beta [float]    --  Set value for beta\n";
#ifdef USE_CUDA
  std::cout << "  --gpu             --  Use available GPUs\n";
#endif
#ifdef USE_MPI
  std::cout << "  --P [uint]        --  Set the process grid row number\n"
               "  --Q [uint]        --  Set the process grid column number\n"
               "  --stat [A/B/C]    --  Set the stationary matrix\n"
               "  --redux           --  Use MPI reductions\n";
#endif
}

int main(int argc, char **argv)
{
 [[maybe_unused]] int ret, rank, size;
 [[maybe_unused]] double start, end;
#ifdef USE_MPI
  int token = 0;
  int mpi_init = MPI_THREAD_SERIALIZED;
#endif
  arg_parser parser(argc, argv);

  if (parser.get("--help"))
  {
    printHelp();
    exit(0);
  }

  [[maybe_unused]] auto type = parser.get<char>("--type", 's');     // Data type used for computations
  [[maybe_unused]] auto m = parser.get<unsigned>("--m", 1024);      // Length of the M dimension for matrices A and C
  [[maybe_unused]] auto n = parser.get<unsigned>("--n", 1024);      // Length of the N dimension for matrices B and C
  [[maybe_unused]] auto k = parser.get<unsigned>("--k", 1024);      // Length of the K dimension for matrices A and B
  [[maybe_unused]] auto bs = parser.get<unsigned>("--bs", 256);     // Length of dimensions for the tiles
  [[maybe_unused]] auto transA = parser.get<char>("--transA", 'N'); // Value of transposition for A, can be 'N' or 'T'
  [[maybe_unused]] auto transB = parser.get<char>("--transB", 'N'); // Value of transposition for B, can be 'N' or 'T'
  [[maybe_unused]] auto alpha = parser.get<float>("--alpha", 1.0);  // Value for the alpha parameter (note : C = alpha * A * B + beta * C)
  [[maybe_unused]] auto beta = parser.get<float>("--beta", 0.0);    // Value for the beta parameter
#ifdef USE_CUDA
  [[maybe_unused]] auto enable_gpu = parser.get("--gpu"); // If activated, use available GPUs
#endif
#ifdef USE_MPI
  [[maybe_unused]] auto process_rows = parser.get<unsigned>("--P", 1); // MPI process grid row size
  [[maybe_unused]] auto process_cols = parser.get<unsigned>("--Q", 1); // MPI process grid column size
  [[maybe_unused]] auto stat = parser.get<char>("--stat", 'C');        // Stationary matrix
#endif
[[maybe_unused]] auto redux = parser.get("--redux");                 // If activated, use MPI reductions
[[maybe_unused]] auto dag = parser.get("--dag");                     // If activated use Heriarchical DAG
  // TODO : Initialize StarPU / MPI + StarPU-MPI / CUBLAS
#ifdef STARPU_OPENMP
struct starpu_parallel_worker_config *parallel_workers;
#endif
#ifdef USE_MPI
  ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
  //std::cout << "Je suis le process " << rank << "\n";
  //std::cout << "Je suis de taille " << size << "\n"; 
  if (size < 2)
  {
    if (rank == 0)
    {
      std::cout << "Need 2 process to work\n";
      starpu_mpi_shutdown();
    }
    if (!mpi_init)
      MPI_Finalize();
    return -1;
  }


#else
  ret = starpu_init(NULL);
#endif

int disk = starpu_disk_register(&starpu_disk_unistd_ops, (void*) "disk_tmp_mpi", 4096 * 4096 * 10);
#ifdef STARPU_OPENMP  
  parallel_workers = starpu_parallel_worker_init(HWLOC_OBJ_SOCKET,
						       STARPU_PARALLEL_WORKER_POLICY_NAME, "dmdas",
						       STARPU_PARALLEL_WORKER_PARTITION_ONE,
						       STARPU_PARALLEL_WORKER_NEW,
						       STARPU_PARALLEL_WORKER_TYPE, STARPU_PARALLEL_WORKER_OPENMP,
//						       STARPU_PARALLEL_WORKER_TYPE, STARPU_PARALLEL_WORKER_INTEL_OPENMP_MKL,
						       STARPU_PARALLEL_WORKER_NB, 2,
						       STARPU_PARALLEL_WORKER_NCORES, 1,
                               0);
#endif
#ifdef STARPU_SIMGRID
  start = starpu_timing_now();
#endif
{
     Matrix<float> matrix1(m, k, bs, bs);
     Matrix<float> matrix2(k, n, bs, bs);
     Matrix<float> result(m, n, bs, bs);
     std::cout << "Matrix 1\n";
     matrix1.fill_value(1);
     std::cout << "\nMatrix 2\n";
     matrix2.fill_value(2);
     starpu_task_wait_for_all();
     result.fill_value(0);
     starpu_task_wait_for_all();
#ifdef USE_MPI
     starpu_mpi_barrier(MPI_COMM_WORLD);
#endif
     // matrix1._tiles[0].print();
     //matrix1.print();
     //matrix2.print();
     //matrix1.assert_equals(matrix2);
     std::cout << "Gemm step\n";
     Matrix<float>::gemm(alpha, matrix1, transA, matrix2, transB, beta, result, redux, dag);
     std::cout << "Finish Gemm step\n";
    //result.print();
    starpu_task_wait_for_all();
    result.print();
    //std::cout << "Finish Task\n";
    //Matrix<float> test(SIZE, SIZE, SIZE/TILENUMBER, SIZE/TILENUMBER);
    //test.fill_value(128);
    //result.assert_equals(test);
    //starpu_task_wait_for_all();
    //test.print();
}
#ifdef STARPU_SIMGRID
  end = starpu_timing_now();
#endif
#ifdef STARPU_OPENMP
starpu_parallel_worker_shutdown(parallel_workers);
#endif
#ifdef USE_MPI
    ret = starpu_mpi_shutdown();
#else
  starpu_shutdown();
#endif
  return 0;
}
