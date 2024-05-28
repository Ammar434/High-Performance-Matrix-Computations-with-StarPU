/**
 * Tutorial: Complete the parts marked TODO in the source files
 * 1) Scaling a vector using 1 task
 * 2) Scaling a vector using a tiling and multiple tasks
 *
 * To compile the program:
 * .c files -> gcc $(pkg-config --cflags starpu-1.4) -c file.c
 * .cu files -> nvcc $(pkg-config --cflags starpu-1.4) -c file.cu
 * .o files -> nvcc $(pkg-config --libs starpu-1.4) file1.o file2.o [...] -o scale
 *
 * To go further:
 * - Look at htop when the program runs. What happens? Why?
 * - Test some environment variables of StarPU e.g.
 *   STARPU_WORKER_STATS STARPU_NCPU STARPU_NCUDA
 *   (execute with ENVIRONMENT_VARIABLE=VALUE ./scale)
 * - Test replacing malloc with starpu_malloc
 */

#include <starpu.h>

#define NX 2048
#define PAR 64

extern void vector_scal_cpu(void *buffers[], void *_args);
extern void vector_scal_cuda(void *buffers[], void *_args);
extern void scal_cpu_func(void *buffers[], void *_args);

static struct starpu_perfmodel perfmodel = {
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "vector_scal"};

static struct starpu_codelet cl = {
	// TODO : Set the codelet functions for cpu and cuda, and it's data parameters
	.cpu_funcs = {vector_scal_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {vector_scal_cuda},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &perfmodel,
};

int main(void)
{
	float *vector;
	double start_time;
	unsigned i;

	// TODO : Initialize StarPU with default configuration
	int ret = starpu_init(NULL);

	vector = malloc(sizeof(vector[0]) * NX);
	for (i = 0; i < NX; i++)
		vector[i] = 1.0f;

	fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);

	starpu_data_handle_t vector_handle;

	// TODO : Register data with StarPU

	float factor = 3.14;

	starpu_vector_data_register(&vector_handle, 0, (uintptr_t)vector_handle, NX, sizeof(vector[0]));

	start_time = starpu_timing_now();

	// TODO : Insert necessary tasks

	ret = starpu_task_insert(&cl,
							 STARPU_VALUE, &factor, sizeof(factor),
							 STARPU_RW, vector_handle,
							 0);

	// TODO : Wait for tasks completion
	starpu_task_wait_for_all();

	fprintf(stderr, "computation took %fµs\n", starpu_timing_now() - start_time);

	// TODO : Unregister the data

	starpu_data_unregister(vector_handle);

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);

	starpu_data_handle_t vector_handles[PAR];

	// TODO : Compute length for sub-vectors (note : we suppose a remainder of 0)
	int len = NX / PAR;

	// TODO : Register all data handles with StarPU

	for (int i = 0; i < PAR; i++)
	{
		starpu_vector_data_register(&vector_handles[i], STARPU_MAIN_RAM, (uintptr_t)(vector + len * i),
									len, sizeof(vector[0]));
	}

	start_time = starpu_timing_now();

	// TODO : Insert necessary tasks for parallel computation
	for (int i = 0; i < PAR; i++)
	{
		starpu_task_insert(&cl,
						   STARPU_VALUE, &factor, sizeof(factor),
						   STARPU_RW, vector_handles[i],
						   0);
	}
	// TODO : Wait for tasks completion

	starpu_task_wait_for_all();

	fprintf(stderr, "computation took %fµs\n", starpu_timing_now() - start_time);

	// TODO : Unregister all data handles

	for (int i = 0; i < PAR; i++)
	{
		starpu_data_unregister(vector_handles[i]);
	}

	fprintf(stderr, "AFTER PARALLEL First element is %f\n", vector[0]);
	free(vector);

	// TODO : Terminate StarPU
	starpu_shutdown();

	return 0;
}
