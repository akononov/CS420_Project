#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=32
#PBS -N cs420_project
#PBS -j oe

cd $PBS_O_WORKDIR

NUM_RANKS=1
#MATRIX_SIZE=100
#BLOCK_SIZE=25

echo "Compiling matrix_product_test.c"
#mpiicc main.c -std=c99 -lrt -qopenmp -o main -g -O0
gcc matrix_product_test.c -std=c99 -lrt -openmp -D_POSIX_C_SOURCE=199309L -lm -o matrix_product_test -O3

echo "Running matrix_product_test"
#mpirun -np ${NUM_RANKS} -ppn 1 valgrind -v --leak-check=yes ./main -t 12 -n ${MATRIX_SIZE} -b ${BLOCK_SIZE}

for N in 128 256 512 1024
	do
	for T in 16 32 64 128 
		do
		for i in 0 1 2 3 4
			do
			echo "Starting test "$i
			aprun -n ${NUM_RANKS} -d 32 ./matrix_product_test -n $N -b $T -t 32
		done
	done
done
