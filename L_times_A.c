#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* These functions compute the product of a lower triangular matrix and a regular dense matrix. They will be used to multiply
the inverse of a diagonal block of L (produced by invert_L.c) by an off-diagonal block of A and thereby solve for an off-diagonal
block of U. These functions will be executed on a single processor using OpenMP for parallelization.*/

void L_A(float* L, float* A, float* product, int M, int N) {
  // product is MxN
  // L is MxM
  // A is MxN

  // iterate over entries of product
  int i, j ,k;
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;  // initialize
      // iterate along row of L/column of A
      for (k=0; k<fmin(i+1,M); k++) {
       product[i*N+j] += L[i*M+k]*A[k*N+j];   // add L[i,k]*A[k,j]
      }
    }
  }
}

void compressedL_A(float* L, float* A, float* product, int M, int N) {
	// product is MxN
  // L is MxM
  // A is MxN
  
  // iterate over entries of product
  int i, j, k, l;
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;	// initialize
      l=i*(i+1)/2;			// start of row i
      // iterate along row of L/column of A
			for (k=0; k<fmin(i+1,M); k++) {
				product[i*N+j] += L[l]*A[k*N+j];
				l++;
			}
		}
	}
}
