#!/usr/bin/env bash

cd "${PBS_O_WORKDIR}"


echo "HEAD_NODE_IP: $HEAD_NODE_IP"

# Load modules as needed (example)
module load singularity

# -------- Image & bind paths (renamed for Singularity) ---------------
singularity_image_path="/home/users/ntu/tianle00/scratch/cache/docker_images/verl_nscc.sif"
IMAGE="/home/users/ntu/tianle00/scratch/cache/docker_images/verl_nscc.sif"
BIND_WORK="$PWD"                                    # project code
SINGULARITY_EXE_PATH="${BIND_WORK}/ray"                                  # ray scripts

export SINGULARITY_CACHEDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/
export SINGULARITY_TMPDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/tmp
export APPTAINER_CACHEDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/
export APPTAINER_TMPDIR=/home/users/ntu/tianle00/scratch/cache/docker_images/.sif_work/tmp

mkdir -p "$SINGULARITY_EXE_PATH"

BIND_DATA="/home/users/ntu/tianle00/scratch/workshop_demo/tmp_cache/"
SARGS="--nv --cleanenv --no-home --env HOME=/root \
  --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root \
  --env HOME=/root,HF_HOME=/root/verl/models/.hf,HF_HUB_CACHE=/root/verl/models/hub,HF_DATASETS_CACHE=/root/verl/models/datasets,TRANSFORMERS_CACHE=/root/verl/models \
  --pwd /workspace"


# --------------------------- CLUSTER VARS ----------------------------
mapfile -t nodes_array < <(uniq "${PBS_NODEFILE}")
NNODES="${NNODES:-${#nodes_array[@]}}"


port="${port:-6379}"                  # Ray GCS port
dashboard_port="${dashboard_port:-8265}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

export RAY_ADDRESS="http://${HEAD_NODE_IP}:${dashboard_port}"

echo "[PBS] Nodes: ${nodes_array[*]}"
echo "[PBS] Head: (${HEAD_NODE_IP})  NNODES=${NNODES}"
echo "[PBS] Ports: GCS=${port}  Dashboard=${dashboard_port}"
echo "[PBS] Nodes and their IPs:"
uniq $PBS_NODEFILE | while read host; do
    echo "$host -> $(getent hosts $host | awk '{print $1}')"
done

# find out time to keep the Ray worker online
WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
SECONDS=`echo $WALLTIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }'`

# --------------------------- START RAY WORKERS -----------------------
cat > "${SINGULARITY_EXE_PATH}/start_worker_in_container.sh" <<EOF
#!/usr/bin/env bash

# ray stop --force

export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray start --address='${HEAD_NODE_IP}:${port}' \
    --num-gpus '${GPUS_PER_NODE}' \
    --disable-usage-stats 

echo "SLEEPING FOR $SECONDS s, to keep Ray worker up"
sleep $SECONDS
EOF

chmod +x "${SINGULARITY_EXE_PATH}/start_worker_in_container.sh"

#  pbsdsh needs to be run in the background (with & at the end) with sleep duration throughout the job to avoid closing the ray cluster when pbsdsh exits: https://stackoverflow.com/questions/72583725/how-to-convert-a-script-that-uses-ssh-to-pbsdsh-while-using-ray      
worker_num=$(( NNODES - 1 ))
if (( worker_num > 0 )); then
  for ((i = 0; i <= worker_num; i++)); do
    node_i="${nodes_array[$i]}"
    echo "[PBS] Starting WORKER $i at ${node_i}"
    pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_worker_in_container.sh"   " &
  done
fi

sleep 20



# ---- quick ray health check (from head) ---- 
cat > "${SINGULARITY_EXE_PATH}/quick_health_check.sh" <<EOF
#!/usr/bin/env bash

ray status || true
EOF

chmod +x "${SINGULARITY_EXE_PATH}/quick_health_check.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
 bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/quick_health_check.sh" "


# pause point to check ray cluster set up
# singularity shell $SARGS "$IMAGE"



# --------------------------- (YOUR TRAINING) -------------------------

cat > "${SINGULARITY_EXE_PATH}/run_training.sh" <<EOF
#!/usr/bin/env bash

export RAY_ADDRESS="http://${HEAD_NODE_IP}:${dashboard_port}"
# Make the project workspace the working directory Ray ships to workers
export WORKING_DIR="/workspace"
export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"


bash recipe/dapo/run_dapo_qwen2.5_0.5b_2gpu_2nodes.sh
EOF

chmod +x "${SINGULARITY_EXE_PATH}/run_training.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
 bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/run_training.sh" "



# --------------------------- CLEANUP RAY -----------------------------
cat > "${SINGULARITY_EXE_PATH}/clean_ray.sh" <<EOF
#!/usr/bin/env bash

ray stop --force || true
EOF

chmod +x "${SINGULARITY_EXE_PATH}/clean_ray.sh"

echo "[PBS] Stopping Ray on all nodes..."

worker_num=$(( NNODES - 1 ))
if (( worker_num > 0 )); then
  for ((i = 0; i <= worker_num; i++)); do
    node_i="${nodes_array[$i]}"
    echo "[PBS] Cleaning up Ray at ${node_i}"
    pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/clean_ray.sh"   " 
  done
fi

echo "[PBS] Job complete."
