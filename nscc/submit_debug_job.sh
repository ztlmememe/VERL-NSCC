#!/bin/sh
## Lines which start with #PBS are directives for the scheduler
## The following line requests the resources for 1 gpu, ngpus=4 for 4 GPUs.

#PBS -l select=1:ngpus=1

## Run for 1 hour, modify as required
#PBS -l walltime=02:00:00

## Submit to correct queue for AI cluster access
#PBS -q normal

## Specify project ID
#PBS -P 12004167

## Job name
#PBS -N submit_debug_job

## Merge standard output and error from PBS script
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/submit_debug_job.log

# Change to directory where job was submitted
set -xeuo pipefail
cd "$PBS_O_WORKDIR" || exit $?

# cd /scratch/experiments/workspace/agents/verl
module load singularity

IMAGE="/home/users/ntu/guoweia3/scratch/images/verl_nscc.sif"
BIND_WORK="$PWD"
BIND_DATA="/home/users/ntu/guoweia3/scratch/experiments/data/agents"

echo Test Complete!