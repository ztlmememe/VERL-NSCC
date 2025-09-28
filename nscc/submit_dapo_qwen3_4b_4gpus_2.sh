#!/bin/sh
## Lines which start with #PBS are directives for the scheduler
## The following line requests the resources for 1 gpu, ngpus=4 for 4 GPUs.

#PBS -l select=1:ngpus=4

## Run for 1 hour, modify as required
#PBS -l walltime=02:00:00

## Submit to correct queue for AI cluster access
#PBS -q normal

## Specify project ID
#PBS -P 12004167

## Job name
#PBS -N dapo_qwen3_4b_4gpus_2

## Merge standard output and error from PBS script
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/create_dapo_qwen3_4b_4gpus_2_train_prompt_bsz4.log

# Change to directory where job was submitted
set -xeuo pipefail
cd "$PBS_O_WORKDIR" || exit $?

# cd /scratch/experiments/workspace/agents/verl
module load singularity

IMAGE="/home/users/ntu/guoweia3/scratch/images/verl_nscc.sif"
BIND_WORK="$PWD"
BIND_DATA="/home/users/ntu/guoweia3/scratch/experiments/data/agents"

# stop Ray on job exit
trap 'singularity exec --nv --cleanenv --no-home \
  --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
  --pwd /workspace "$IMAGE" bash -lc "ray stop --force || true"' EXIT

# run inside container
singularity exec --nv --cleanenv --no-home \
  --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
  --pwd /workspace "$IMAGE" /bin/bash nscc/run_insidecon_dapo_qwen3_4b_4gpus_2.sh
