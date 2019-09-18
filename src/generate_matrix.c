#include <stdio.h>
#include <stdlib.h>

/* Generate a random MxN matrix */

void generate_matrix(float* matrix, int M, int N) {
	int i,j;
	#pragma omp parallel for
	for (i=0; i<M; i++) {
		for (j=0; j<N; j++) {
			matrix[i*N+j]=(float)rand() / (float)RAND_MAX;
		}
	}
}
