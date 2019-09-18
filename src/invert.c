#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* These functions compute the inverses of triangular matrices. They will be used to invert the diagnal blocks of L and U,
which is the first step in solving for the off-diagonal blocks of U and L (respectively). The functions will be executed
on a single processor using OpenMP for parallelization.*/

void invert_L(float* L, float* inverse, int N) {
	// L, inverse are NxN
	float temp_sum;
	int i,j,k;
	for(j=0; j<N; j++) {
		inverse[j*N+j]=1;	// inv[j,j] = 1
		for(i=j+1; i<N; i++) {
			temp_sum=0;
			for(k=j;k<i;k++) {
				temp_sum-=L[i*N+k]*inverse[k*N+j];
			}
			inverse[i*N+j]=temp_sum;
		}
	}
}

void invert_U(float* U, float* inverse, int N) {
	// U, inverse are NxN
	float temp_sum;
	int i,j,k;
	for (j=0; j<N; j++) {
		inverse[j*N+j]=1/U[j*N+j];	// inv[j,j] = 1/U[j,j]
		for (i=j-1; i>-1; i--) {
			temp_sum=0;
			for(k=j; k>i; k--) {
				temp_sum-=U[i*N+k]*inverse[k*N+j];
			}
			inverse[i*N+j]= temp_sum/U[i*N+i];
		}
	}
}
