#!/usr/bin/env bash

cd "${PBS_O_WORKDIR}"

# Load modules as needed (example)
module load singularity

# -------- Image & bind paths (renamed for Singularity) ---------------
singularity_image_path="/home/users/ntu/guoweia3/scratch/images/verl_nscc.sif"
IMAGE="/home/users/ntu/guoweia3/scratch/images/verl_nscc.sif"
BIND_WORK="$PWD"                                    # project code
SINGULARITY_EXE_PATH="${BIND_WORK}/ray"                                  # ray scripts

mkdir -p "$SINGULARITY_EXE_PATH"

BIND_DATA="/home/users/ntu/guoweia3/scratch/experiments/data/agents"  # shared data
SARGS="--nv --cleanenv --no-home --env HOME=/root \
  --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root \
  --pwd /workspace"


# --------------------------- CLUSTER VARS ----------------------------
mapfile -t nodes_array < <(uniq "${PBS_NODEFILE}")
NNODES="${NNODES:-${#nodes_array[@]}}"

head_node="${nodes_array[0]}"
head_node_ip="$(getent hosts "${head_node}" | awk '{print $1; exit}')"


port="${port:-6379}"                  # Ray GCS port
dashboard_port="${dashboard_port:-8265}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

export RAY_ADDRESS="http://${head_node_ip}:${dashboard_port}"

echo "[PBS] Nodes: ${nodes_array[*]}"
echo "[PBS] Head: ${head_node} (${head_node_ip})  NNODES=${NNODES}"
echo "[PBS] Ports: GCS=${port}  Dashboard=${dashboard_port}"



# --------------------------- START RAY HEAD --------------------------
WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
SECONDS=`echo $WALLTIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }'`
echo "Keeping nodes' Ray execution in background for $SECONDS s"


cat > "${SINGULARITY_EXE_PATH}/start_head_in_container.sh" <<EOF
#!/usr/bin/env bash

export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray stop --force

ray start --head \
    --node-ip-address='${head_node_ip}' \
    --port='${port}' \
    --dashboard-host=0.0.0.0 --dashboard-port='${dashboard_port}' \
    --num-gpus '${GPUS_PER_NODE}' \
    --disable-usage-stats 

echo "SLEEPING FOR $SECONDS s, to keep Ray cluster up"
sleep $SECONDS
EOF

chmod +x "${SINGULARITY_EXE_PATH}/start_head_in_container.sh"


# pbsdsh needs to be run in the background (with & at the end) with sleep duration throughout the job to avoid closing the ray cluster when pbsdsh exits: https://stackoverflow.com/questions/72583725/how-to-convert-a-script-that-uses-ssh-to-pbsdsh-while-using-ray      
pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_head_in_container.sh"   " &

sleep 10


# --------------------------- START RAY WORKERS -----------------------
cat > "${SINGULARITY_EXE_PATH}/start_worker_in_container.sh" <<EOF
#!/usr/bin/env bash


export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray start --address='${head_node_ip}:${port}' \
    --num-gpus '${GPUS_PER_NODE}' \
    --disable-usage-stats 


echo "SLEEPING FOR $SECONDS s, to keep Ray worker up"
sleep $SECONDS
EOF

chmod +x "${SINGULARITY_EXE_PATH}/start_worker_in_container.sh"

#  pbsdsh needs to be run in the background (with & at the end) with sleep duration throughout the job to avoid closing the ray cluster when pbsdsh exits: https://stackoverflow.com/questions/72583725/how-to-convert-a-script-that-uses-ssh-to-pbsdsh-while-using-ray      
worker_num=$(( NNODES - 1 ))
if (( worker_num > 0 )); then
  for ((i = 1; i <= worker_num; i++)); do
    node_i="${nodes_array[$i]}"
    echo "[PBS] Starting WORKER $i at ${node_i}"
    pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_worker_in_container.sh"   " &
  done
fi

sleep 10



# ---- quick ray health check (from head) ---- 
cat > "${SINGULARITY_EXE_PATH}/quick_health_check.sh" <<EOF
#!/usr/bin/env bash

ray status || true
echo hello inside health1
EOF

chmod +x "${SINGULARITY_EXE_PATH}/quick_health_check.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
 bash -lc "module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/quick_health_check.sh" "


# pause point to check ray cluster set up
# singularity shell $SARGS "$IMAGE"



# --------------------------- (YOUR TRAINING) -------------------------

cat > "${SINGULARITY_EXE_PATH}/run_training.sh" <<EOF
#!/usr/bin/env bash

export RAY_ADDRESS="http://${head_node_ip}:${dashboard_port}"
# Make the project workspace the working directory Ray ships to workers
export WORKING_DIR="/workspace"
export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

bash recipe/dapo/run_dapo_qwen3_4b_2nodes4A100.sh
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
