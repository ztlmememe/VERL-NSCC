#!/bin/sh
## Lines which start with #PBS are directives for the scheduler
## The following line requests the resources for 1 gpu, ngpus=4 for 4 GPUs.

#PBS -l select=1:ncpus=128:mem=512gb

## Run for 1 hour, modify as required
#PBS -l walltime=12:00:00

## Submit to correct queue for AI cluster access
#PBS -q normal

## Specify project ID
#PBS -P 12004167

## Job name
#PBS -N create_singularity_image

## Merge standard output and error from PBS script
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/create_singularity_image.log

# Change to directory where job was submitted
cd "$PBS_O_WORKDIR" || exit $?

module load singularity

pwd 

singularity build verl_nscc.sif docker://hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0
