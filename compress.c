#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* These functions convert triangular matrices to and from compressed format. Lower triangular matrices are stored row-wise,
while upper triangular matrices are stored column-wise.*/

void compress_L(float* L, float* packed, int M, int N) {
	// L is M x N
	int i,j,k=0;
	for (i=0; i<M; i++) {
		for (j=0; j<fmin(N,i+1); j++) {
			packed[k]=L[i*N+j];
			k++;
		}
	}
}

void compress_U(float* U, float* packed, int M, int N) {
	// U is MxN
	int i,j,k=0;
	for (j=0; j<N; j++) {
		for (i=0;i<fmin(j+1,M); i++) {
			packed[k]=U[i*N+j];
			k++;
		}
	}
}

void uncompress_L(float* packed, float* L, int M, int N) {
	// L is MxN
	int i,j,k=0;
	for (i=0; i<M; i++) {
		for (j=0; j<fmin(N,i+1); j++) {
			L[i*N+j]=packed[k];
			k++;
		}
	}
}

void uncompress_U(float* packed, float* U, int M, int N) {
	// U is MxN
	int i,j,k=0;
	for (j=0; j<N; j++) {
		for (i=0; i=fmin(j+1,M); i++) {
			U[i*N+j]=packed[k];
			k++;
		}
	}
}
