#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* These functions compute the product of a lower triangular matrix and a regular dense matrix. They will be used to multiply
the inverse of a diagonal block of L (produced by invert_L.c) by an off-diagonal block of A and thereby solve for an off-diagonal
block of U. These functions will be executed on a single processor using OpenMP for parallelization.*/

void L_A(float* L, float* A, float* product, int M, int N) {
  // product is MxN
  // L is MxM
  // A is MxN

  // iterate over entries of product
  int i, j ,k;
//  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {	
      product[i*N+j]=0;  // initialize
      // iterate along row of L/column of A
      for (k=0; k<i; k++) {
       product[i*N+j] += L[i*M+k]*A[k*N+j];   // add L[i,k]*A[k,j]
      }
      product[i*N+j] += A[i*N+j]; // add L[i,i]*A[i,j]=A[i,j]
    }
  }
}


void compressedL_A(float* L, float* A, float* product, int M, int N) {
	// product is MxN
  // L is MxM
  // A is MxN
  
  // iterate over entries of product
  int i, j, k, l;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;	// initialize
      l=i*(i-1)/2;			// start of row i
      // iterate along row of L/column of A
			for (k=0; k<i; k++) {
				product[i*N+j] += L[l]*A[k*N+j]; // add L[i,k]*A[k,j]
				l++;
			}
			product[i*N+j] += A[i*N+j]; // add L[i,i]*A[i,j]=A[i,j]
		}
	}
}



/* These functions compute the product of a dense matrix and an upper triangular matrix. They will be used to multiply
the inverse of a diagonal block of U (produced by invert_U.c) by an off-diagonal block of A and thereby solve for an off-diagonal
block of L. These functions will be executed on a single processor using OpenMP for parallelization.*/

void A_U(float* A, float* U, float* product, int M, int N) {
  // product is MxN
  // A is MxN
  // U is NxN

  // iterate over entries of product
  int i, j, k;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;  // initialize
			// iterate along row of A/column of U
      for (k=0; k<j+1; k++) {
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
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;	// initialize
      u=j*(j+1)/2;			// start of col j 
      // iterate along row of A/column of U
			for (k=0; k<j+1; k++) {
				product[i*N+j] += A[i*N+k]*U[u];
				u++;
			}
		}
	}
}



/* This function computes the product of two dense matrices (L and U blocks) and then subtracts that product from
a dense matrix (A block). It will be used to update blocks of A after we've solved for a pair of off-diagonal L and U
blocks whose product contributes to that block of A. The function will be executed on a single processor using OpenMP
for parallelization.*/

void AmLU(float* A, float* L, float* U, int M, int N, int K) {
  // A is MxN
  // L is MxK
  // U is KxN

  // iterate over entries of A
  int i, j ,k;
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      // iterate along row of L/column of U
      for (k=0; k<K; k++) {
       A[i*N+j] -= L[i*K+k]*U[k*N+j];   // subtract L[i,k]*U[k,j]
      }
    }
  }
}
