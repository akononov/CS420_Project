#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "generate_matrix.c"
#include "L_times_A.c"
#include "compress.c"


void A_U(float* A, float* U, float* product, int M, int N);
void A_compressedU(float* A, float* U, float* product, int M, int N);
void L_A(float* L, float* A, float* product, int M, int N);
void compressedL_A(float* L, float* A, float* product, int M, int N);
void generate_matrix(float* matrix, int M, int N);
void compress_L(float* L, float* compressed, int N);
void compress_U(float* U, float* compressed, int N);
void uncompress_L(float* compressed, float* L, int N);
void uncompress_U(float* compressed, float* U, int N);

int main (int argc, char** argv) {
	int N=100;
	float* matrix = (float*)malloc(sizeof(float)*N*N);
	float* A = (float*)malloc(sizeof(float)*N*N);
	float* U = (float*)malloc(sizeof(float)*N*N);
	float* L = (float*)malloc(sizeof(float)*N*N);
	float* compressed = (float*)malloc(sizeof(float)*N*N);
	float* product = (float*)malloc(sizeof(float)*N*N);
  struct timespec start_time, end_time;
  struct timespec total_start, total_end;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_REALTIME, &total_start);
  clock_gettime(CLOCK_REALTIME, &start_time);

generate_matrix(matrix, N, N);

clock_gettime(CLOCK_REALTIME, &end_time);
  double init_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
printf("generate matrix time: %f\n", init_time);

generate_matrix(U, N, N);//generate upper matrix, not trangular
generate_matrix(L, N, N);//generate lower matrix, not trangular
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_REALTIME, &total_start);
  clock_gettime(CLOCK_REALTIME, &start_time);

A_U(matrix, U, product, N, N);

clock_gettime(CLOCK_REALTIME, &end_time);
   init_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
printf("A*U time: %f\n", init_time);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_REALTIME, &total_start);
  clock_gettime(CLOCK_REALTIME, &start_time);

compress_L(L, compressed, N);

clock_gettime(CLOCK_REALTIME, &end_time);
   init_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
printf("compress L time: %f\n", init_time);

	free(matrix);
	free(A);
	free(U);
	free(L);
	free(compressed);
	free(product);
	
return 0;

}
