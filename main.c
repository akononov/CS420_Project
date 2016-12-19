#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "util.c"
#include "generate_matrix.c"
#include "compress.c"
#include "LU.c"
#include "invert.c"
#include "matrix_product.c"

int main(int argc, char** argv){

	// Begin timing
	struct timespec start_time, end_time;
	clock_gettime(CLOCK_REALTIME, &start_time);

	// Parse command line arguments
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
	if (myrank==0)
		printf("Tasked with decomposing a %d x %d matrix partitioned into %d x %d blocks using %d processes and %d threads per process\n",
  												N, N, block_size, block_size, size, num_threads);
  
/*	if (provided != MPI_THREAD_FUNNELED) {
		if (myrank==0) {
			printf("Error: Requested thread support '%d', but only received '%d'\n", required, provided);
			return 1;
		}
	}
*/
	// Set and check thread number (default: 32)
	omp_set_num_threads(num_threads);
	if (myrank==0) {
		#pragma omp parallel
		{
			if (omp_get_thread_num() == 0)
				printf("Running with %lu OpenMP threads.\n", omp_get_num_threads());
		}
	}
  
	// Create row/column ring communicators on 2D torus
	MPI_Comm TORUS_COMM, ROW_COMM, COL_COMM;
	int mycoords[2];
	int periods[2]={1,1};		// periodic
	int reorder=1;			// allow reordering
	int dims[2]={0,0};		// let MPI set dimensions
	MPI_Dims_create(size,2,dims);
	if (myrank==0)
		printf("Setting up a %dx%d torus communicator\n",dims[0],dims[1]);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &TORUS_COMM);	// make torus
	MPI_Cart_coords(TORUS_COMM, myrank, 2, mycoords);				// get my coordinates
	MPI_Comm_split(TORUS_COMM, mycoords[0], mycoords[1], &ROW_COMM);		// make row ring
	MPI_Comm_split(TORUS_COMM, mycoords[1], mycoords[0], &COL_COMM);		// make column ring
  
	// Clear the cache
//	clear_cache();
  
	// ===== Allocate memory =====
	float* A = (float*)malloc(sizeof(float)*block_area);
	float* Inverses = (float*)malloc(sizeof(float)*block_area); // L[n][n]^(-1) and U[n][n]^(-1)

	// estimate number of L, U matrices per process
	int estLUcount = n_blocks/size;
	int estLUsize = estLUcount*block_area;
	int rowLsize = estLUsize*dims[1];
	int colUsize = estLUsize*dims[0];
  
	float* compressed_Linv = (float*)malloc(sizeof(float)*block_size*(block_size-1)/2);
	float* compressed_Uinv = (float*)malloc(sizeof(float)*block_size*(block_size+1)/2);
	float* myLs = (float*)malloc(sizeof(float)*estLUsize);	// L blocks I compute
	float* myUs = (float*)malloc(sizeof(float)*estLUsize);	// U blocks I compute
	float* rowLs = (float*)malloc(sizeof(float)*rowLsize);	// buffers for gathering L, U
	float* colUs = (float*)malloc(sizeof(float)*colUsize);
	if (myrank==0) {
		printf("estLUsize: %d floats\n",estLUsize);
		printf("Size of rowLs: %d B\n",sizeof(float)*estLUsize*dims[1]);
		printf("Size of rowUs: %d B\n",sizeof(float)*estLUsize*dims[0]);
	}
  
	// initialize more variables/arrays
	int givemework=1;
	int task;
	int myLUcount, myLUsize;
	int allLsizes[dims[1]], allUsizes[dims[0]], allLdisps[dims[1]], allUdisps[dims[0]];
	int totLsize, totUsize;
	allLdisps[0]=0;
	allUdisps[0]=0;
  
  
	// ========= Iterate over stages ==============
	for (size_t n=0; n<n_blocks-1; n++){		// will want a minimum block...
		myLUcount=0;
		myLUsize=0;
		task=1;
  	
  	
		// ========= MASTER =========
		if (myrank==0) {
			printf("Starting stage %d\n",n);

			// LU decomposition of A[n][n]
			generate_matrix(A, block_size, block_size);
			inplace_LU(A, block_size);
			
			// Invert L[n][n] and U[n][n]
			invert_L(A, Inverses, block_size);
			invert_U(A, Inverses, block_size);
			
			// Broadcast L^-1[n][n] and U^-1[n][n]
			MPI_Bcast(Inverses, block_area, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// Allocate to slaves computation of U[n][task] and L[task][n]
			int num_slaves = size-1;
			MPI_Status status;
			for (task=n+1; task<n_blocks; task++) {
				// get work request
				MPI_Recv(&givemework, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				// send task to process that requested work
				MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
				printf("Sent task %d to process %d\n",task,status.MPI_SOURCE);
			}
			
			// Tell slaves that we're done
			task=0;
			while (num_slaves>0) {
				// get work request
				MPI_Recv(&givemework, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				// send "done" message to process that requested work
				MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
				printf("Sent task %d to process %d\n",task,status.MPI_SOURCE);
				num_slaves--;
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
				printf("process %d received task %d\n",myrank,task);

				if (task != 0) {
					// start new request for work
					MPI_Isend(&givemework, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
					MPI_Irecv(&task, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
					
					// increment my count and check for sufficient memory
					myLUcount++;
					if (myLUcount > estLUcount) {
						// expand myLs and myUs
						size_t new_size=sizeof(float)*(myLUcount+2)*block_area;
						myLs = (float*)realloc(myLs, new_size);
						myUs = (float*)realloc(myUs, new_size);
						printf("Process %d reallocated myLs and myUs\n",myrank);
						printf("New size: %d floats\n", new_size/sizeof(float));
					}
					
					// compute L[task][n]
					generate_matrix(A, block_size, block_size);
					compressedL_A_tiled(compressed_Linv, A, myLs+myLUsize, block_size, block_size);
					// compute U[n][task]
					generate_matrix(A, block_size, block_size);
					A_compressedU_tiled(A, compressed_Uinv, myUs+myLUsize, block_size, block_size);
					myLUsize += block_area;
					
					printf("process %d has completed %d tasks\n",myrank,myLUcount);
				}
			}
		}

		// ========= EVERYONE ==========
	  
		// Gather L counts from row and U counts from column
		MPI_Request gather[2];
		MPI_Iallgather(&myLUsize, 1, MPI_INT, allLsizes, 1, MPI_INT, ROW_COMM, &gather[0]);
		MPI_Iallgather(&myLUsize, 1, MPI_INT, allUsizes, 1, MPI_INT, COL_COMM, &gather[1]);
		MPI_Waitall(2, gather, MPI_STATUSES_IGNORE);			// wait for counts
		
		printf("process %d has myLsize %d and received L sizes %d, %d\n",myrank,myLUsize,allLsizes[0],allLsizes[1]);
		printf("process %d has myUsize %d and received U sizes %d, %d\n",myrank,myLUsize,allUsizes[0],allUsizes[1]);
	  
		// Compute total counts and displacements of L and U blocks
		for (int i=1; i<dims[1]; i++) {
			allLdisps[i] = allLdisps[i-1] + allLsizes[i-1];
		}
		for (int i=1; i<dims[0]; i++) {
			totUsize+=allUsizes[i];
			allUdisps[i] = allUdisps[i-1] + allUsizes[i-1];
		}
		totLsize=allLdisps[dims[1]-1]+allLsizes[dims[1]-1];
		totUsize=allUdisps[dims[0]-1]+allUsizes[dims[0]-1];
			
		// Check for sufficient memory
		printf("process %d has totLsize %d, totUsize %d and computed L displacements %d, %d, and U displacements %d, %d\n", myrank, totLsize, totUsize,allLdisps[0],allLdisps[1],allUdisps[0],allUdisps[1]);
		if (totLsize > rowLsize) {
			rowLs = (float*)realloc(rowLs, sizeof(float)*totLsize);
			rowLsize = totLsize;
			printf("Process %d reallocated rowLs to %d floats\n",myrank,totLsize);
		}
		if (totUsize > colUsize) {
			colUs = (float*)realloc(colUs, sizeof(float)*totUsize);
			colUsize = totUsize;
			printf("Process %d reallocated colUs to %d floats\n",myrank,totUsize);
		}
	  
		// Gather L[i][n] from row and U[n][j] from column
		MPI_Request gatherv[2];
		printf("process %d is sending %d floats in allgatherv\n", myrank, myLUcount*block_area);
		MPI_Iallgatherv(myLs, myLUsize, MPI_FLOAT, rowLs, allLsizes, allLdisps, MPI_FLOAT, ROW_COMM, &gatherv[0]);
		MPI_Iallgatherv(myUs, myLUsize, MPI_FLOAT, colUs, allUsizes, allUdisps, MPI_FLOAT, COL_COMM, &gatherv[1]);
		printf("started allgathers!\n");
	
		if (myrank != 0) {	
			// update A[i][j] using all of my L[i][n], U[n][j]
			for (int l=0; l<myLUsize; l+=block_area) {
				for (int u=0; u<myLUsize; u+=block_area) {
					generate_matrix(A, block_size, block_size);
					AmLU_tiled(A, myLs+l, myUs+u, block_size, block_size, block_size);
				}
			}
			printf("process %d: done computing my LU pairs\n",myrank);
		}


		// update A[i][j] using all received L[i][n], U[n][j]
		MPI_Waitall(2, gatherv, MPI_STATUSES_IGNORE);
		for (int col=0; col<dims[1]; col++) {
			for (int row=0; row<dims[0]; row++) {
				if (row != mycoords[0] || col != mycoords[1]) { // already computed with my pairs;
					for (int l=allLdisps[col]; l<allLdisps[col]+allLsizes[col]; l+=block_area) {
						for (int u=allUdisps[row]; u<allUdisps[row]+allUsizes[row]; u+=block_area) {
							generate_matrix(A, block_size, block_size);
							AmLU_tiled(A, rowLs+l, colUs+u, block_size, block_size, block_size);
						}
					}
				}
			}
		}
		printf("process %d: done computing all LU pairs\n",myrank);
	}
  
  	// LU decomposition of final block
  	if (myrank==0) {
		generate_matrix(A, block_size, block_size);
		inplace_LU(A, block_size);
		printf("Reached end of computation!\n");
	}
  
	// free memory and finalize

	printf("process %d freeing A\n",myrank);
	free(A);
	printf("process %d freeing Inverses\n",myrank);
	free(Inverses);
	printf("process %d freeing Linv\n",myrank);
	free(compressed_Linv);
	printf("process %d freeing Uinv\n",myrank);
	free(compressed_Uinv);
	printf("process %d freeing myL\n",myrank);
	free(myLs);
	printf("process %d freeing myUs\n",myrank);
	free(myUs);
	printf("process %d freeing rowLs\n",myrank);
	free(rowLs);
	printf("process %d freeing colUs\n",myrank);
	free(colUs);


	MPI_Finalize();
	
	// end timing
	clock_gettime(CLOCK_REALTIME, &end_time);
	double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
	printf("Total execution time: %f\n", run_time);

	return 0;
}
