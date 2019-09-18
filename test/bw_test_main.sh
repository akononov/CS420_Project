#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l nodes=2:ppn=32
#PBS -N cs420_project
#PBS -j oe

cd $PBS_O_WORKDIR

TILE_SIZE=128

echo "Compiling main.c"
#mpiicc main.c -std=c99 -lrt -qopenmp -o main -g -O0
cc main.c -lrt -openmp -D_POSIX_C_SOURCE=199309L -lm -o main -O3

echo "Running main"
#mpirun -np ${NUM_RANKS} -ppn 1 valgrind -v --leak-check=yes ./main -t 12 -n ${MATRIX_SIZE} -b ${BLOCK_SIZE}

for NUM_RANKS in 2 4 8
do
	for MATRIX_SIZE in 1024 2048 4098 8192
	do
		for BLOCK_SIZE in 256 512 1024
		do
			THREADS=$(expr 64 / $NUM_RANKS)
			echo "BASH: Running with "$NUM_RANKS" ranks, "$THREADS" threads, "$MATRIX_SIZE" matrix, and "$BLOCK_SIZE" block"
			aprun -n $NUM_RANKS -d $THREADS ./main -n $MATRIX_SIZE -b $BLOCK_SIZE -d $THREADS -t $TILE_SIZE
		done
	done
done
