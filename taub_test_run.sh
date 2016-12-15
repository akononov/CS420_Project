#!/bin/bash

#PBS -l walltime=00:30:00
#PBS -l nodes=2:ppn=12
#PBS -N cs420_project
#PBS -q cse
#PBS -j oe

cd $PBS_O_WORKDIR

module load valgrind

NUM_RANKS=2
MATRIX_SIZE=100
BLOCK_SIZE=25

echo "Running small matrix"
mpirun -np ${NUM_RANKS} -ppn 1 valgrind -v --leak-check=yes ./main -t 12 -n ${MATRIX_SIZE} -b ${BLOCK_SIZE}


