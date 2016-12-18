#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "generate_matrix.c"


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
  #pragma omp parallel for schedule(guided)
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      product[i*N+j]=0;		// initialize
      l=i*(i-1)/2;		// start of row i
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
  # pragma omp parallel for schedule(guided)
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
  # pragma omp parallel for schedule(guided)
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

void AmLU_tiled(float* A, float* L, float* U, int M, int N, int K, int T) {
  // A is MxN
  // L is MxK
  // U is KxN
  // T=tile size

  int i, j ,k, ii, jj, kk;
  float* temp_sum = (float*)malloc(sizeof(float)*M*N*K/T);
//  float temp_sum[M*N*K/T];
  
  // iterate over tiles
  #pragma omp parallel for schedule(guided)
  for (ii=0; ii<M/T; ii++) {
    for (jj=0; jj<N/T; jj++) {
      for (kk=0; kk<K/T; kk++) {
	// iterate over entries of A within tile
        for (i=ii*T; i<(ii+1)*T; i++) {
          for (j=jj*T; j<(jj+1)*T; j++) {
            temp_sum[(i*N+j)*K/T+kk]=0;
            // iterate along row of L/column of U
            for (k=kk*T; k<(kk+1)*T; k++) {
              temp_sum[(i*N+j)*K/T+kk] += L[i*K+k]*U[k*N+j];   // subtract L[i,k]*U[k,j]
            }
          }
      	} 
      }
    }
  }

  # pragma omp parallel for schedule(guided)  
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      for (kk=0; kk<K/T; kk++) {
        A[i*N+j]-=temp_sum[(i*N+j)*K/T+kk];
      }
    }	
  }

  free(temp_sum);
}


void main(){
	int N=4000, T=32;
	float* A1 = (float*)malloc(sizeof(float)*N*N);
	float* A2 = (float*)malloc(sizeof(float)*N*N);
	float* L = (float*)malloc(sizeof(float)*N*N);
	float* U = (float*)malloc(sizeof(float)*N*N);

	generate_matrix(A1,N,N);
	generate_matrix(L,N,N);
	generate_matrix(U,N,N);
	int i, j;
	for (i=0; i<N*N; i++)
		A2[i]=A1[i];
	
	struct timespec start_time, end_time;
	clock_gettime(CLOCK_REALTIME, &start_time);
	AmLU(A1,L,U,N,N,N);
	clock_gettime(CLOCK_REALTIME, &end_time);
	double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Regular time: %f\n", run_time);
	
	clock_gettime(CLOCK_REALTIME, &start_time);
	AmLU_tiled(A2,L,U,N,N,N,T);
	clock_gettime(CLOCK_REALTIME, &end_time);
	run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Tiled time: %f\n", run_time);
	

/*
	for (i=0; i<N; i++){
		for (j=0; j<N; j++){
			printf("%f, ",A1[i*N+j]);
		}
		printf("\t");
		for (j=0; j<N; j++){
			printf("%f, ",A2[i*N+j]);
		}
		printf("\n");
	}	
	printf("\n");
*/
	
	free(A1);
	free(A2);
	free(L);
	free(U);
}

