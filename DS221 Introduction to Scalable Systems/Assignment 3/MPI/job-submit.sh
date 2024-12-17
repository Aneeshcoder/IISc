#!/bin/bash
#SBATCH --job-name main
#SBATCH --tasks-per-node 32
#SBATCH --nodelist node[4-7]

module load openmpi

cd $SLURM_SUBMIT_DIR
make

# Loop over different process counts (1, 8, 16, 32, 64)
for num_processes in 32 #1 8 16 32 64
do
  echo "Running with $num_processes processes"
  mpiexec -n $num_processes ./main
done