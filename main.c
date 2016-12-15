#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "util.h"

int main(int argc, char** argv){

	// Parse commandline arguments
  size_t N, num_threads, block_size;
  parse_args(argc, argv, &N, &num_threads, &block_size);
  size_t n_blocks=N/block_size;
  size_t block_area=block_size*block_size;

	// Initialize MPI
	int required = MPI_THREAD_FUNNELED, provided;
	int myrank, size;
	MPI_Init_thread(&argc, &argv, required, &provided);	// not sure what happens with argc, argv
	
	// Get rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (myrank==0)	printf("Tasked with decomposing a %d x %d matrix partitioned into %d x %d blocks using %d processes and %d threads per process",
  												N, N, block_size, bloc_size, size, num_threads);
  
	if (provided != MPI_THREAD_FUNNELED) {
		if (myrank==0) {
			printf("Error: Requested thread support '%d', but only received '%d'\n", required, provided);
			return 1;
		}
	}

	// Set and check thread number (default: 32)
  omp_set_num_threads(num_threads);
  if (myrank==0) {
	  #pragma omp parallel
		{
  	  if (omp_get_thread_num() == 0) {
  	    printf("Running with %lu OpenMP threads.\n", omp_get_num_threads());
  	  }
  	}
  }
  
  // Create row/column ring communicators on 2D torus
  MPI_Comm TORUS_COMM, MPI_Comm ROW_COMM, MPI_Comm COL_COMM;
  int mycoords[2];
	int periods[2]={1,1};	// periodic
	int reorder=1;				// allow reordering
	int dims[2]={0,0};		// let MPI set dimensions
	MPI_Dims_create(size,2,dims);
	if (rank==0)
		printf("Setting up a %dx%d torus communicator",dims[0],dims[1]);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &TORUS_COMM);	// make torus
	MPI_Cart_coords(TORUS_COMM, myrank, 2, mycoords);													// get my coordinates
	MPI_Comm_split(TORUS_COMM, mycoords[0], mycoords[1], &ROW_COMM);					// make row ring
  MPI_Comm_split(TORUS_COMM, mycoords[1], mycoords[0], &COL_COMM);					// make column ring
  
  // Clear the cache
//  clear_cache();
  
  // Begin timing
//  struct timespec start_time, end_time;
//  clock_gettime(CLOCK_REALTIME, &start_time);
  
  
  // ===== Allocate memory =====
  float* A = (float*)malloc(sizeof(float)*block_area);
  float* Inverses = (float*)malloc(sizeof(float)*block_area); // L[n][n]^(-1) and U[n][n]^(-1)
  // estimate number of L, U matrices per process
	int	estLUcount = n_blocks/(size-1)*1.1;
	long estLUsize = estLUcount*block_area;
  
  // master
  if (myrank==0) {
		float* myLs, myUs;	// empty; needed for collective communication
	}
	
	// slaves
	else {
		float* compressed_Linv = (float*)malloc(sizeof(float)*block_size*(block_size-1)/2);
		float* compressed_Uinv = (float*)malloc(sizeof(float)*block_size*(block_size+1)/2);
  	float* myLs = (float*)malloc(sizeof(float)*estLUsize);	// L blocks I compute
  	float* myUs = (float*)malloc(sizeof(float)*estLUsize);	// U blocks I compute
  }
	float* rowLs = (float*)malloc(sizeof(float)*estLUsize*dims[1]);	// buffers for gathering L, U
	float* colUs = (float*)malloc(sizeof(float)*estLUsize*dims[0]);
  
  // initialize more variables/arrays
  int givemework=1;
  size_t task=1;
  size_t myLUcount, myLUindex;
  size_t rowLcounts[dims[1]], colUcounts[dims[0]], rowLdisps[dims[1]], colUdisps[dims[0]];
  rowLdisps[0]=0;
  colUdisps[0]=0;
  
  
  // ========= Iterate over stages ==============
  for (size_t n=0; n<n_blocks; n++){		// will want a minimum block...
  	myLUcount=0;
  	myLUindex=0;
  	
  	
	  // ========= MASTER =========
		if (myrank==0) {

			// LU decomposition of A[n][n]
			generate_matrix(Ann, block_size, block_size);
			inplace_LU(Ann, block_size);
			
			// Invert L[n][n] and U[n][n]
			invert_L(Ann, Inverses, block_size);
			invert_U(Ann, Inverses, block_size);
			
			// Broadcast L^-1[n][n] and U^-1[n][n]
			MPI_Bcast(Inverses, block_area, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// Allocate to slaves computation of U[n][task] and L[task][n]
			int num_slaves = size-1;
			MPI_Status status;
			for (task=n+1, task<n_blocks, k++) {
				// get work request
				MPI_Recv(&givemework, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				// send task to process that requested work
				MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
			}
			
			// Tell slaves that we're done
			task=0;
			while (num_slaves>0) {
				// get work request
				MPI_Recv(&givemework, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				// send "done" message to process that requested work
				MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
				numslaves--;
			}
		} 
  
  
		// ========= SLAVES =========
		else {
		
			// receive L^-1[n][n] and U^-1[n][n]
			MPI_Bcast(Inverses, block_area, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
			// fetch work
			MPI_Request req[2];
			MPI_Isend(&givemework, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
			MPI_Irecv(&task, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
			
			// compress L[n][n]^-1 and U[n][n]^-1
			compress_L(Inverses, compressed_Linv, block_size);
			compress_U(Inverses, compressed_Uinv, block_size);
		
			// until done
			while (task != 0) {
			
				// wait for task
				MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

				if (task != 0) {
					// start new request for work
					MPI_Isend(&givemework, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
					MPI_Irecv(&task, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
					
					// increment my count and check for sufficient memory
					myLUcount++;
					if (myLUcount > estLUcount) {
						// expand myLs and myUs
						estLUcount = myLUcount*1.1;
						estLUsize = estLUcount*block_area;
						myLs = (float*)realloc(myLs, sizeof(float)*estLUsize);
						myUs = (float*)realloc(myUs, sizeof(float)*estLUsize);
					}
					
					// compute L[task][n]
					generate_matrix(A, block_size, block_size);
					compressedL_A(compressed_Linv, A, &myLs[myLUindex], block_size, block_size);
					// compute U[n][task]
					generate_matrix(A, block_size, block_size);
					A_compressedU(A, compressed_Uinv, &myUs[myLUindex], block_size, block_size);
					myLUindex += block_area;
				}
			}
		}

	  // ========= EVERYONE ==========
	  
	  // Gather L counts from row and U counts from column
	  MPI_Request gather[4];
	  MPI_Iallgather(myLcount*block_area, 1, MPI_INT, rowLcounts, 1, MPI_INT, ROW_COMM, &gather[0]);
	  MPI_Iallgather(myUcount*block_area, 1, MPI_INT, colUcounts, 1, MPI_INT, COL_COMM, &gather[1]);
	  
	  // Compute displacements of L and U blocks
	  for (int i=1, i<dims[1], i++)
	  	rowLdisps[i] = rowLdisps[i-1] + rowLcounts[i-1];
	  for (int i=1, i<dims[0], i++)
	  	colUdisps[i] = colUdisps[i-1] + colUcounts[i-1];
	  
		// Gather L[i][n] from row and U[n][j] from column
		MPI_Waitall(2, gather, MPI_STATUSES_IGNORE);			// wait for counts
		MPI_Iallgatherv(myLs, myLcount*block_area, MPI_FLOAT, rowLs, rowLcounts, rowLdisps, MPI_FLOAT, ROW_COMM, &gather[2]);
		MPI_Iallgatherv(myUs, myIcount*block_area, MPI_FLOAT, colUs, colUcounts, colUdisps, MPI_FLOAT, COL_COMM, &gather[3]);
		
		// update A[i][j] using all of my L[i][n], U[n][j]
		
		// COMPUTE...
		for (int l=0; l<myLcount; l++)
			for (int u=0; u<myUcount; u++) {
				
			}
		}
		
		// update A[i][j] using all received L[i][n], U[n][j]
		MPI_Waitall(2, &gather[2], MPI_STATUSES_IGNORE);
		
		// COMPUTE
		
	}
  
  // finalize and free memory
	MPI_Finalize();
	free(Inverses);
  if (myrank==0)
  	free(Ann);
	else {
		free(compressed_Linv);
		free(compressed_Uinv);
		free(myLs);
		free(myUs);
  }
  free(rowLs);
  free(colUs);
	
  // end timing
//  clock_gettime(CLOCK_REALTIME, &end_time);
//  double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Total time: %f\n", run_time);
}
