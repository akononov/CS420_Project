#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "util.h"

#define BLOCK_SIZE 1024

int main(int argc, char** argv){

	// Parse commandline arguments
  size_t N, num_threads;
  parse_args(argc, argv, &N, &num_threads);
  size_t n_blocks=N/BLOCK_SIZE;

	// Initialize MPI
	int required = MPI_THREAD_FUNNELED;
	int provided;
	int myrank, size;
	
	MPI_Init_thread(&argc, &argv, required, &provided);
  
	if (provided != MPI_THREAD_FUNNELED) {
		printf("Error: Requested thread support '%d', but only received '%d'\n", required, provided);
		return 1;
	}

	// Set and check thread number (default: 32)
  omp_set_num_threads(num_threads);
  #pragma omp parallel
	{
    if (omp_get_thread_num() == 0) {
      printf("Running with %lu OpenMP threads.\n", omp_get_num_threads());
    }
  }

	// Get rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Create row/column ring communicators on 2D torus
  MPI_Comm TORUS_COMM, MPI_Comm ROW_COMM; MPI_Comm COL_COMM;
  int mycoords[2];
	int periods[2]={1,1};	// periodic
	int reorder=1;				// allow reordering
	int dims[2]={0,0};		// let MPI set dimensions
	MPI_Dims_create(size,2,dims);
	if (rank==0)
		printf("Setting up a %dx%d torus communicator",dims[0],dims[1]);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &TORUS_COMM);
	MPI_Cart_coords(TORUS_COMM, myrank, 2, mycoords);
	MPI_Comm_split(TORUS_COMM, mycoords[0], mycoords[1], &ROW_COMM);
  MPI_Comm_split(TORUS_COMM, mycoords[1], mycoords[0], &COL_COMM);
  
  // Clear the cache
  clear_cache();
  
  // Begin timing
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_REALTIME, &start_time);
  
  // Allocate memory
  float* Inverses = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE);
  int	estLUcount = n_blocks/(size-1)*1.1;
  if (myrank==0) {
		float* Ann = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE);
		float* myLs, myUs;
	}
	else {
		float* compressed_Linv = (float*)malloc(sizeof(float)*BLOCK_SIZE*(BLOCK_SIZE-1)/2);
		float* compressed_Uinv = (float*)malloc(sizeof(float)*BLOCK_SIZE*(BLOCK_SIZE+1)/2);
  	float* myLs = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE*estLUcount);
  	float* myUs = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE*estLUcount);
  }
	float* rowLs = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE*estLUcount*dims[1]);
	float* colUs = (float*)malloc(sizeof(float)*BLOCK_SIZE*BLOCK_SIZE*estLUcount*dims[0]);
  
  // Initialize variables/arrays used repeatedly
  int myLcount, myUcount, rowLcounts[dims[1]], colUcounts[dims[0]], rowLdisps[dims[1]], colUdisps[dims[0]];
  rowLdisps[0]=0;
  colUdisps[0]=0;
  
  // Iterate over stages
  for (n=0; n<n_blocks; n++){
  	myLcount=0;
		myUcount=0;
  	
	  // ========= MASTER =========
		if (myrank==0) {

			// LU decomposition of A[n][n]
			generate_matrix(Ann, BLOCK_SIZE, BLOCK_SIZE);
			inplace_LU(Ann, BLOCK_SIZE);
			
			// Invert L[n][n] and U[n][n]
			invert_L(Ann, Inverses, BLOCK_SIZE);
			invert_U(Ann, Inverses, BLOCK_SIZE);
			
			// Broadcast L^-1[n][n] and U^-1[n][n]
			MPI_Bcast(Inverses, BLOCK_SIZE*BLOCK_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// Allocate to slaves U[n][j]=L^-1[n][n]*A[n][j] and L[i][n]=A[i][n]*U^-1[n][n]
			int num_slaves = size-1;
			MPI_Status status;
			while (num_slaves>0) {
				// get work request
				MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

				// if not done
				if (task[0] < n_blocks) {
					// send task
					MPI_Send(task, 2, MPI_UINT64_T, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
					
					// increment task
					if (task[1] < n_blocks-1) {
						task[1]++;
					}
					else {
						task[0]++;
						task[1]=task[0]; // fix to account for diagonal blocks...
					}
				}
				// no more work to send
				else {
					numslaves--;
					MPI_Send(task, 2, MPI_UINT64_T, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
				}
			}
		} 
  
		// ========= SLAVES =========
		else {
		
			// Receive L^-1[n][n] and U^-1[n][n]
			MPI_Bcast(Inverses, BLOCK_SIZE*BLOCK_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
			// request for work
			MPI_Request req[2];
			MPI_Isend(givemework, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
			MPI_Irecv(task, 2, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &req[1]);
			
			// Compress L^-1[n][n] and U^-1[n][n]
			compress_L(Inverses, compressed_Linv, BLOCK_SIZE);
			compress_U(Inverses, compressed_Uinv, BLOCK_SIZE);
		
			// until done
			while (task[0] < n_blocks) {
				// wait for task
				MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

				if (task[0] < n_blocks) {
					// start new request for work
					MPI_Isend(givemework, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
					MPI_Irecv(task, 2, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &req[1]);
					
					// increment my count and check for sufficient memory
					if (task[]>task[]){
						myLcount+=1;
						if (myLcount > avgLUcount)
							//realloc
					}
					else {
						myUcount+=1;
						if (myUcount > avgLUcount)
							//realloc
					}
					
					// COMPUTE...
				}
			}
		}

	  // ========= EVERYONE =========
	  
	  // Gather L counts from row and U counts from column
	  MPI_Request gather[4];
	  MPI_Iallgather(myLcount*BLOCK_SIZE*BLOCK_SIZE, 1, MPI_INT, rowLcounts, 1, MPI_INT, ROW_COMM, &gather[0]);
	  MPI_Iallgather(myUcount*BLOCK_SIZE*BLOCK_SIZE, 1, MPI_INT, colUcounts, 1, MPI_INT, COL_COMM, &gather[1]);
	  
	  // Compute displacements of L and U blocks
	  for (int i=1, i<dims[1], i++)
	  	rowLdisps[i] = rowLdisps[i-1] + rowLcounts[i-1];
	  for (int i=1, i<dims[0], i++)
	  	colUdisps[i] = colUdisps[i-1] + colUcounts[i-1];
	  
		// Gather L[i][n] from row and U[n][j] from column
		MPI_Waitall(2, gather, MPI_STATUSES_IGNORE);			// wait for counts
		MPI_Iallgatherv(myLs, myLcount*BLOCK_SIZE*BLOCK_SIZE, MPI_FLOAT, rowLs, rowLcounts, rowLdisps, MPI_FLOAT, ROW_COMM, &gather[2]);
		MPI_Iallgatherv(myUs, myIcount*BLOCK_SIZE*BLOCK_SIZE, MPI_FLOAT, colUs, colUcounts, colUdisps, MPI_FLOAT, COL_COMM, &gather[3]);
		
		// update A[i][j] using all of my L[i][n], U[n][j]

		// COMPUTE...
		
		// update A[i][j] using all received L[i][n], U[n][j]
		MPI_Waitall(2, &gather[2], MPI_STATUSES_IGNORE);
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
  clock_gettime(CLOCK_REALTIME, &end_time);
  double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Total time: %f\n", run_time);
}
