#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* This function computes the LU decomposition of a matrix block using Guassian elimination. It will be used
to factor the diagonal blocks of the matrix and it will execute on a single processor using OpenMP for parallelization.
Assume U is initialized with the value of A. */

void LU(float* L, float* U, int N) {
	// A, L, U are NxN
	// assume U is initialized with a copy of A
	
	// iterate over columns to be eliminated
	int i,j,k;
	for (k=0; k<N; k++) {
		L[k*N+k]=1;		// L[k,k]=1
		
		// eliminate column k in rows i>k
		#pragma omp parallel for
		for (i=k+1; i<N; i++) {
			L[i*N+k] = U[i*N+k]/U[k*N+k];		// multiplier
			U[i*N+k] = 0;

			// update row i of U and L
			for (j=k+1; j<N; j++) {
				L[i*N+j] = 0;
				U[i*N+j] -= L[i*N+k] * U[k*N+j];
			}
		}
	}
}

void inplace_LU(float* A, int N) {
	// A will be overwritten with L below the diagonal and U above the diagonal
	// A, L, U are NxN
	
	// iterate over columns to be eliminated
	int i,j,k;
	for (k=0; k<N; k++) {
		// eliminate column k in rows i>k
		#pragma omp parallel for
		for (i=k+1; i<N; i++) {
			A[i*N+k] = A[i*N+k]/A[k*N+k];		// multiplier L[i,k]

			// update row i of U
			for (j=k+1; j<N; j++) {
				A[i*N+j] -= A[i*N+k] * A[k*N+j];
			}
		}
	}
}
