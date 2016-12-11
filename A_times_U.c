#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* These functions compute the product of an upper triangular matrix and a regular dense matrix. They will be used to multiply
the inverse of a diagonal block of U (produced by invert_U.c) by an off-diagonal block of A and thereby solve for an off-diagonal
block of L. These function will be executed on a single processor using OpenMP for parallelization.*/

void A_U(float* A, float* U, float* product, int M, int N) {
  // product is MxN
  // A is MxN
  // U is NxN

  // iterate over entries of product
  int i, j, k;
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;  // initialize
			// iterate along row of A/column of U
      for (k=0; k<fmin(j+1,N); k++) {
       product[i*N+j] += A[i*N+k]*U[k*N+j];   // add A[i,k]*U[k,j]
      }
    }
  }
}


void A_compressedU(float* A, float* U, float* product, int M, int N) {
	// product is MxN
  // A is MxN
  // U is NxN
  
  // iterate over entries of product
  int i, j, k, u;
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;	// initialize
      u=j*(j+1)/2;			// start of col j 
      // iterate along row of A/column of U
			for (k=0; k<fmin(j+1,N); k++) {
				product[i*N+j] += A[i*N+k]*U[u];
				u++;
			}
		}
	}
}
