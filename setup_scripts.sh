
# on NTU GPU Workstation
docker run --rm -it --gpus all  \
 --shm-size=64g   -v "$(pwd)":/workspace   \
 -v /mnt/ssd/alvinchan/experiments/data/agents:/root   \
 -v /mnt/ssd/alvinchan/tmp:/tmp   \
 -w /workspace   \
 hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0 bash



ray start --head
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
# bash recipe/dapo/run_dapo_qwen2.5_32b.sh # or other scripts
bash recipe/dapo/run_dapo_qwen3_4b_ntugpuws.sh


# NSCC cluster uses PBS Pro as the job scheduler.

singularity build verl_nscc.sif docker://hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0

# interactive sesson
qsub -I -l select=1:ngpus=1 -P personal-guoweia3  -l walltime=02:00:00 -q ai
qsub -I -l select=1:ngpus=1 -P personal-guoweia3  -l walltime=02:00:00 -q normal
qsub -I -l select=1:ncpus=4:mem=64gb -l walltime=02:00:00 -q normal 

# check projects' resources
myprojects

# check storage quota
myquota

# check modules with cuda
module avail 2>&1 | grep cuda

