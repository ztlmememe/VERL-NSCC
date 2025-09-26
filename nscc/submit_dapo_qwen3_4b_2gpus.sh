#!/bin/sh
## Lines which start with #PBS are directives for the scheduler
## The following line requests the resources for 1 gpu, ngpus=4 for 4 GPUs.

#PBS -l select=1:ngpus=4

## Run for 1 hour, modify as required
#PBS -l walltime=06:00:00

## Submit to correct queue for AI cluster access
#PBS -q normal

## Specify project ID
#PBS -P 12004167

## Job name
#PBS -N create_singularity_image

## Merge standard output and error from PBS script
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/create_dapo_qwen3_4b_4gpus.%J.log

# Change to directory where job was submitted
cd "$PBS_O_WORKDIR" || exit $?

module load singularity

singularity exec --nv --cleanenv --no-home \
  --bind "$PWD":/workspace,/home/users/ntu/guoweia3/scratch/experiments/data/agents:/root \
  --pwd /workspace \
  /home/users/ntu/guoweia3/scratch/images/verl_nscc.sif \
  bash nscc/run_insidecon_dapo_qwen3_4b_2gpus.sh
