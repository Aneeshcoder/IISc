#!/bin/csh
#SBATCH --job-name main
#SBATCH --tasks-per-node 32
#SBATCH --nodelist node[4-7]

cd $SLURM_SUBMIT_DIR
make
./main