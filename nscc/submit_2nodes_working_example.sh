#!/usr/bin/env bash
#PBS -N dapo_ray_2n4g_sing
#PBS -q normal
#PBS -P 12004167
#PBS -l select=2:ngpus=4
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/dapo_ray_2n4g_sing.log

# set -xeuo pipefail
cd "$PBS_O_WORKDIR"

bash nscc/run_2nodes_working_example.sh

echo submit_2nodes_working_example.sh COMPLETED