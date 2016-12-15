#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "util.c"
#include "generate_matrix.c"
#include "compress.c"
#include "LU.c"
#include "invert.c"
#include "matrix_product.c"

int main(int argc, char** argv){

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
  
	// Begin timing
//	struct timespec start_time, end_time;
//	clock_gettime(CLOCK_REALTIME, &start_time);
  
  
	// ===== Allocate memory =====
	float* A = (float*)malloc(sizeof(float)*block_area);
	float* Inverses = (float*)malloc(sizeof(float)*block_area); // L[n][n]^(-1) and U[n][n]^(-1)

	// estimate number of L, U matrices per process
	int estLUcount = n_blocks/size;
	size_t estLUsize = estLUcount*block_area;
  
	float* compressed_Linv = (float*)malloc(sizeof(float)*block_size*(block_size-1)/2);
	float* compressed_Uinv = (float*)malloc(sizeof(float)*block_size*(block_size+1)/2);
	float* myLs = (float*)malloc(sizeof(float)*estLUsize);	// L blocks I compute
	float* myUs = (float*)malloc(sizeof(float)*estLUsize);	// U blocks I compute
	float* rowLs = (float*)malloc(sizeof(float)*estLUsize*dims[1]);	// buffers for gathering L, U
	float* colUs = (float*)malloc(sizeof(float)*estLUsize*dims[0]);
	if (myrank==0) {
		printf("estLUsize: %d\n",estLUsize);
		printf("Size of rowLs: %d\n",sizeof(float)*estLUsize*dims[1]);
		printf("Size of rowUs: %d\n",sizeof(float)*estLUsize*dims[0]);
	}
  
	// initialize more variables/arrays
	int givemework=1;
	int task;
	int myLUcount, myLUindex;
	int rowLcounts[dims[1]], colUcounts[dims[0]], rowLdisps[dims[1]], colUdisps[dims[0]];
	int totLcount, totUcount;
	rowLdisps[0]=0;
	colUdisps[0]=0;
  
  
	// ========= Iterate over stages ==============
	for (size_t n=0; n<n_blocks; n++){		// will want a minimum block...
		myLUcount=0;
		myLUindex=0;
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
						myLs = (float*)realloc(myLs, (myLUcount+2)*block_area);
						myUs = (float*)realloc(myUs, (myLUcount+2)*block_area);
						printf("Process %d reallocated myLs and myUs\n",myrank);
						printf("New size: %d\n", sizeof(float)*estLUsize);
					}
					
					// compute L[task][n]
					generate_matrix(A, block_size, block_size);
					compressedL_A(compressed_Linv, A, &myLs[myLUindex], block_size, block_size);
					// compute U[n][task]
					generate_matrix(A, block_size, block_size);
					A_compressedU(A, compressed_Uinv, &myUs[myLUindex], block_size, block_size);
					myLUindex += block_area;
					
					printf("process %d has completed %d tasks\n",myrank,myLUcount);
				}
			}
		}

		// ========= EVERYONE ==========
	  
		// Gather L counts from row and U counts from column
		MPI_Request gather[2];
		MPI_Iallgather(&myLUcount, 1, MPI_INT, rowLcounts, 1, MPI_INT, ROW_COMM, &gather[0]);
		MPI_Iallgather(&myLUcount, 1, MPI_INT, colUcounts, 1, MPI_INT, COL_COMM, &gather[1]);
		MPI_Waitall(2, gather, MPI_STATUSES_IGNORE);			// wait for counts
	  
		// Compute total counts and displacements of L and U blocks
		totLcount=rowLcounts[0];
		totUcount=rowUcounts[0];
		for (int i=1; i<dims[1]; i++) {
			totLcount+=rowLcounts[i];
			rowLdisps[i] = rowLdisps[i-1] + rowLcounts[i-1]*block_area;
		}
		for (int i=1; i<dims[0]; i++) {
			totUcount+=colUcounts[i];
			colUdisps[i] = colUdisps[i-1] + colUcounts[i-1]*block_area;
		}
			
		// Check for sufficient memory
		if (totLcount > estLUcount*dims[1]) {
			rowLs = (float*)realloc(rowLs, sizeof(float)*totLcount*block_area);
			printf("Process %d reallocated rowLs\n",myrank);
		}
		if (totUcount > estLUcount*dims[0]) {
			colUs = (float*)realloc(colUs, sizeof(float)*totUcount*block_area);
			printf("Process %d reallocated colUs\n",myrank);
		}
	  
		// Gather L[i][n] from row and U[n][j] from column
		MPI_Request gatherv[2];
		printf("process %d is sending %d floats in allgatherv\n", myrank, myLUcount*block_area);
		MPI_Iallgatherv(myLs, myLUcount*block_area, MPI_FLOAT, rowLs, rowLcounts, rowLdisps, MPI_FLOAT, ROW_COMM, &gatherv[0]);
		MPI_Iallgatherv(myUs, myLUcount*block_area, MPI_FLOAT, colUs, colUcounts, colUdisps, MPI_FLOAT, COL_COMM, &gatherv[1]);
	
		size_t Lindex=0, Uindex=0;
		if (myrank != 0) {	
			// update A[i][j] using all of my L[i][n], U[n][j]
			for (int l=0; l<myLUcount; l++) {
				for (int u=0; u<myLUcount; u++) {
					generate_matrix(A, block_size, block_size);
					AmLU(A, &myLs[Lindex], &myUs[Uindex], block_size, block_size, block_size);
					Uindex+=block_area;
				}
				Uindex=0;
				Lindex+=block_area;
			}
		}

		// update A[i][j] using all received L[i][n], U[n][j]
		MPI_Waitall(2, gatherv, MPI_STATUSES_IGNORE);
		
		for (int col=0; col<dims[1]; col++) {
			Lindex=rowLdisps[col];
			for (int row=0; row<dims[0]; row++) {
				if (row != mycoords[0] || col != mycoords[1]) { // already computed with my pairs
					Uindex=colUdisps[row];
					for (int l=0; l<rowLcounts[col]; l++) {
						for (int u=0; u<colUcounts[row]; u++) {
							generate_matrix(A, block_size, block_size);
							AmLU(A, &rowLs[Lindex], &colUs[Uindex], block_size, block_size, block_size);
							Uindex+=block_area;
						}
					Uindex=colUdisps[row];
					Lindex+=block_area;
					}
				}
			}
		}
	}
  
  // free memory and finalize
	MPI_Finalize();
	free(Inverses);
	free(A);
  if (myrank!=0) {
		free(compressed_Linv);
		free(compressed_Uinv);
		free(myLs);
		free(myUs);
  }
  free(rowLs);
  free(colUs);
	
  // end timing
/*  clock_gettime(CLOCK_REALTIME, &end_time);
  double run_time = (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9 +
                     (double)(end_time.tv_sec - start_time.tv_sec);
  printf("Total time: %f\n", run_time);
  */
}
