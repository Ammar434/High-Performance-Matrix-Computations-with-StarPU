/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2021  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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

#include <starpu.h>

/* This kernel takes a buffer and scales it by a constant factor */
void vector_scal_cpu(void *buffers[], void *cl_arg)
{
	// TODO : Extract vector interface from buffers
	struct starpu_vector_interface *vector = buffers[0];

	// TODO : Extract vector pointer and length
	float *val = (float *)STARPU_VECTOR_GET_PTR(vector);
	unsigned n = STARPU_VECTOR_GET_NX(vector);

	// TODO : Extract the scaling factor
	float factor;
	starpu_codelet_unpack_args(cl_arg, &factor);

	// TODO : Scale the vector
	for (unsigned i = 0; i < n; i++)
		val[i] *= factor;
}
