#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=12
#PBS -N cs420_project
#PBS -q secondary 
#PBS -j oe

cd $PBS_O_WORKDIR

NUM_RANKS=1
MATRIX_SIZE=100
BLOCK_SIZE=25

echo "Compiling matrix_product_test.c"
#mpiicc main.c -std=c99 -lrt -qopenmp -o main -g -O0
mpiicc matrix_product_test.c -std=c99 -lrt -qopenmp -D_POSIX_C_SOURCE=199309L -o matrix_product_test

echo "Running matrix_product_test"
#mpirun -np ${NUM_RANKS} -ppn 1 valgrind -v --leak-check=yes ./main -t 12 -n ${MATRIX_SIZE} -b ${BLOCK_SIZE}

for N in 256 512 1024 2048 4098 8192
	do
	for T in 16 32 64 128 256 512
		do
		mpirun -np ${NUM_RANKS} -ppn 1 ./matrix_product_test -n $N -b $T -t 12
	done
done
