#!/bin/sh
## Lines which start with #PBS are directives for the scheduler
## The following line requests the resources for 1 gpu, ngpus=4 for 4 GPUs.

#PBS -l select=1:ngpus=2

## Run for 1 hour, modify as required
#PBS -l walltime=02:00:00

## Submit to correct queue for AI cluster access
#PBS -q normal

## Specify project ID
#PBS -P 12004167

## Job name
#PBS -N dapo_qwen2.5_0.5b_2gpu

## Merge standard output and error from PBS script
#PBS -j oe
#PBS -o /home/users/ntu/tianle00/scratch/workshop_demo/job_logs/submit_dapo_qwen2.5_0.5b_2gpu.log

# Change to directory where job was submitted cd /home/users/ntu/tianle00/scratch/verl
set -xeuo pipefail
cd "$PBS_O_WORKDIR" || exit $?

# cd /scratch/experiments/workspace/agents/verl
module load singularity

IMAGE="/home/users/ntu/tianle00/scratch/cache/docker_images/verl_nscc.sif"
BIND_WORK="$PWD"
BIND_DATA="/home/users/ntu/tianle00/scratch/workshop_demo/tmp_cache/"

# IMAGE=/home/users/ntu/tianle00/scratch/cache/docker_images/verl_nscc.sif
# HOST_CACHE=/home/users/ntu/tianle00/scratch/cache/verl

export SINGULARITY_CACHEDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/
export SINGULARITY_TMPDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/tmp
export APPTAINER_CACHEDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/
export APPTAINER_TMPDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/tmp



# stop Ray on job exit
# trap 'singularity exec --nv --cleanenv --no-home \
#   --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
#   --pwd /workspace "$IMAGE" bash -lc "ray stop --force || true"' EXIT


trap 'singularity exec --nv --cleanenv --no-home \
  --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
  --env HOME=/root,HF_HOME=/root/verl/models/.hf,HF_HUB_CACHE=/root/verl/models/hub,HF_DATASETS_CACHE=/root/verl/models/datasets,TRANSFORMERS_CACHE=/root/verl/models \
  --pwd /workspace "$IMAGE" bash -lc "ray stop --force || true"' EXIT



# run inside container
# singularity exec --nv --cleanenv --no-home \
#   --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
#   --pwd /workspace "$IMAGE" /bin/bash nscc/run_insidecon_dapo_qwen3_4b_4gpus_2.sh
singularity exec --nv --cleanenv --no-home \
  --bind "$BIND_WORK":/workspace,"$BIND_DATA":/root \
  --env HOME=/root,HF_HOME=/root/verl/models/.hf,HF_HUB_CACHE=/root/verl/models/hub,HF_DATASETS_CACHE=/root/verl/models/datasets,TRANSFORMERS_CACHE=/root/verl/models \
  --pwd /workspace "$IMAGE" \
  /bin/bash nscc/run_insidecon_dapo_qwen2.5_0.5b_2gpu.sh

