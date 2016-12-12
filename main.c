#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "util.h"

#define BLOCK_SIZE 1024

int main(int argc, char** argv){

	// Initialize MPI
	int required = MPI_THREAD_FUNNELED;
	int provided;
	int myrank, size;
	
  MPI_Init_thread(&argc, &argv, required, &provided);
  
  if (provided != MPI_THREAD_FUNNELED) {
    printf("Error: Requested thread support '%d', but only received '%d'\n", required, provided);
    return 1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  // Create Cartesian communicator (2 by size/2)
  
  // Parse commandline arguments
  size_t n, num_threads;
  parse_args(argc, argv, &n, &num_threads);
  size_t n_blocks=n/BLOCK_SIZE;
  size_t task[2] = {0,0};		// starting block
  int req;
  int *blocks = (int*)malloc(sizeof(int) * n_blocks * n_blocks);

  // Set the number of threads (default: 32)
  omp_set_num_threads(num_threads);

  // Set up OpenMP
	#pragma omp parallel
	{
    if (omp_get_thread_num() == 0) {
      printf("Running with %lu OpenMP threads.\n", omp_get_num_threads());
    }
  }
  
  // Clear the cache
  clear_cache();
  
  // Begin timing
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_REALTIME, &start_time);
  
  for (layer=0; layer<n_blocks; layer++){
  
	  // ========= MASTER =========
	  if (myrank==0) {

			int num_slaves = size-1;
			MPI_Status status;
     
	    // until all slaves have finished
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
				else {
					numslaves--;
					MPI_Send(task, 2, MPI_UINT64_T, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
				}
			}
	  }
  
  
  
  // ========= SLAVE 1: diagonal blocks =========
  else if (rank==1) {
  
  }
  
  
  
  // ========= OTHER SLAVES: off diagonal blocks =========
  else {
  	// send work request
		MPI_Send(&req, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		// receive task
		MPI_Recv(task, 2, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// until done
		while (task[0] < n_blocks) {

			// COMPUTE...
			
			// send work request
			MPI_Send(&req, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			// receive task
			MPI_Recv(task, 2, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		MPI_Finilize();
  }
  
  
  
  // End timing
  clock_gettime(CLOCK_REALTIME, &end_time);
  double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Total time: %f\n", run_time);
}

int get_task(size_t *task, process, task_matrix) 
