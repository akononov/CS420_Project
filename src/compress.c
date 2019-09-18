#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* These functions convert triangular matrices to and from compressed format. Lower triangular matrices are stored row-wise,
while upper triangular matrices are stored column-wise. The diagonal entries of lower triangular matrices are not stored
because they are always 1.*/

void compress_L(float* L, float* compressed, int N) {
	// L is NxN
	int i,j,k;
	#pragma omp parallel for schedule(guided)
	for (i=1; i<N; i++) {
		k=i*(i-1)/2;    // starting index of row i
		for (j=0; j<i; j++) {
			compressed[k]=L[i*N+j];
			k++;
		}
	}
}

void compress_U(float* U, float* compressed, int N) {
	// U is NxN
	int i,j,k;
	#pragma omp parallel for schedule(guided)
	for (j=0; j<N; j++) {
	  k=j*(j+1)/2;
		for (i=0; i<j+1; i++) {
			compressed[k]=U[i*N+j];
			k++;
		}
	}
}

void uncompress_L(float* compressed, float* L, int N) {
	// L is NxN
	int i,j,k;
	#pragma omp parallel for schedule(guided)
	for (i=0; i<N; i++) {
	  k=i*(i-1)/2;
	  L[i*N+i]=1;  // L[i,i]=1
		for (j=0; j<i; j++) {
			L[i*N+j]=compressed[k];
			k++;
		}
	}
}

void uncompress_U(float* compressed, float* U, int N) {
	// U is NxN
	int i,j,k;
	#pragma omp parallel for schedule(guided)
	for (j=0; j<N; j++) {
	  k=j*(j+1)/2;
		for (i=0; i<j+1; i++) {
			U[i*N+j]=compressed[k];
			k++;
		}
	}
}
