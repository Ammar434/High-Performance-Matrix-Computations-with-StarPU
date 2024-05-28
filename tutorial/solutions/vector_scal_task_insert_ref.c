/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2012, 2014, 2019, 2021-2022  Université de Bordeaux 1
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to submit a task to StarPU
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */
#include <starpu.h>

#define NX 2048
#define PAR 64

extern void vector_scal_cpu(void *buffers[], void *_args);
extern void vector_scal_cuda(void *buffers[], void *_args);
extern void vector_scal_opencl(void *buffers[], void *_args);

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
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector,
								NX, sizeof(vector[0]));

	float factor = 3.14;

	start_time = starpu_timing_now();

	// TODO : Insert necessary tasks
	ret = starpu_task_insert(&cl,
							 STARPU_VALUE, &factor, sizeof(factor),
							 STARPU_RW, vector_handle,
							 0);

	// TODO : Wait for tasks completion
	starpu_task_wait_for_all();

	fprintf(stderr, "computation took %fµs\n", starpu_timing_now() - start_time);

	// TODO : Unregister data
	starpu_data_unregister(vector_handle);

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

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);
	free(vector);

	// TODO : Terminate StarPU
	starpu_shutdown();

	return 0;
}
