#!/usr/bin/env bash
# --------------------------- PBS DIRECTIVES ---------------------------
#PBS -N verl-ray-on-pbs
#PBS -q your-queue
#PBS -P your-account
#PBS -l select=2:ncpus=64:ngpus=4:mem=200gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o $HOME/slurm-$PBS_JOBID.out

# --------------------------- RUNTIME SETUP ---------------------------
# set -euo pipefail
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
# RAY_TMP="/home/users/ntu/guoweia3/scratch/tmp/ray-${PBS_JOBID}"           # per-job ray temp

# Common Singularity args (mirrors original)
# SARGS="--nv --cleanenv --no-home --env HOME=/root \
#   --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root,${RAY_TMP}:/raytmp \
#   --pwd /workspace"
SARGS="--nv --cleanenv --no-home --env HOME=/root \
  --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root \
  --pwd /workspace"
# SARGS="--nv --cleanenv --no-home --env HOME=/root \
#   --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root,${TMPDIR}:/tmp \
#   --pwd /workspace"


# SINGULARITY_SARGS="--nv --cleanenv --no-home --env HOME=/root \
#   --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root,${TMPDIR}:/tmp"

# --------------------------- CLUSTER VARS ----------------------------
mapfile -t nodes_array < <(uniq "${PBS_NODEFILE}")
NNODES="${NNODES:-${#nodes_array[@]}}"

head_node="${nodes_array[0]}"
head_node_ip="$(getent hosts "${head_node}" | awk '{print $1; exit}')"

# # debug head node ip
# DEV=$(ip route show default | awk '{print $5; exit}')
# head_node_ip=$(ip -4 addr show "$DEV" | awk '/inet /{sub(/\/.*/,"",$2); print $2; exit}')



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
echo "SLEEPING FOR $SECONDS s"


cat > "${BIND_WORK}/ray/start_head_in_container.sh" <<EOF
#!/usr/bin/env bash

export WORKING_DIR="/workspace"

export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ls 

ray stop --force

echo ray arg1 inside head ${head_node_ip}:${port}
echo ray arg2 inside head ${dashboard_port}

ray start --head \
    --node-ip-address='${head_node_ip}' \
    --port='${port}' \
    --dashboard-host=0.0.0.0 --dashboard-port='${dashboard_port}' \
    --num-gpus '${GPUS_PER_NODE}' \
    --disable-usage-stats 

echo "SLEEPING FOR $SECONDS s"
sleep $SECONDS

EOF

chmod +x "${BIND_WORK}/ray/start_head_in_container.sh"

# singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_head_in_container.sh"

# pbsdsh -n 0 -- \
#   singularity exec --nv --bind $verl_workdir $apptainer_image_path \
#     ray start --head \
#     --node-ip-address='${head_node_ip}' \
#     --port='${port}' \
#     --dashboard-host=0.0.0.0 --dashboard-port='${dashboard_port}' \
#     --num-gpus '${GPUS_PER_NODE}' \
#     --disable-usage-stats 


# SINGULARITY_SARGS="--nv \
#   --bind ${BIND_WORK}:/workspace"

# pbsdsh -n 0 -- singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_head_in_container.sh" 


# pbsdsh needs to be run in the background (with & at the end) with sleep duration throughout the job to avoid closing the ray cluster when pbsdsh exits: https://stackoverflow.com/questions/72583725/how-to-convert-a-script-that-uses-ssh-to-pbsdsh-while-using-ray      
pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "echo \$SARGS && module load singularity && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_head_in_container.sh"   " &

# pbsdsh -n 0 -- bash -lc "
# singularity exec $SARGS '${singularity_image_path}' \
#     ray start --head \
#       --node-ip-address='${head_node_ip}' \
#       --port='${port}' \
#       --dashboard-host=0.0.0.0 --dashboard-port='${dashboard_port}' \
#       --num-gpus '${GPUS_PER_NODE}' \
#       --disable-usage-stats 
# "

sleep 10

# --------------------------- START RAY WORKERS -----------------------
cat > "${BIND_WORK}/ray/start_worker_in_container.sh" <<EOF
#!/usr/bin/env bash

echo ray arg1 inside worker ${HEAD_IP}:${GCS_PORT}
echo ray arg2 inside worker ${OBJECT_MANAGER_PORT}
echo ray arg3 inside worker ${NODE_MANAGER_PORT}

export WORKING_DIR="/workspace"

export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ls 

ray start --address='${head_node_ip}:${port}' \
    --num-gpus '${GPUS_PER_NODE}' \
    --disable-usage-stats 


sleep $SECONDS

EOF

chmod +x "${BIND_WORK}/ray/start_worker_in_container.sh"

#  pbsdsh needs to be run in the background (with & at the end) with sleep duration throughout the job to avoid closing the ray cluster when pbsdsh exits: https://stackoverflow.com/questions/72583725/how-to-convert-a-script-that-uses-ssh-to-pbsdsh-while-using-ray      
worker_num=$(( NNODES - 1 ))
if (( worker_num > 0 )); then
  for ((i = 1; i <= worker_num; i++)); do
    node_i="${nodes_array[$i]}"
    echo "[PBS] Starting WORKER $i at ${node_i}"
    pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
    bash -lc "echo \$SARGS && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_worker_in_container.sh"   " &
  done
fi

sleep 10




# pbsdsh -n 1 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#     bash -lc "echo \$SARGS && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/start_worker_in_container.sh"   " &


# pbsdsh -n 1 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#     bash -lc "echo \$SARGS && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "ls /tmp"   "
# pbsdsh -n 1 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#     bash -lc "echo \$SARGS && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "pwd /tmp"   "


# ---- quick health check (from head) ---- WIP
cat > "${BIND_WORK}/ray/quick_health_check.sh" <<EOF
#!/usr/bin/env bash

ray status || true
echo hello inside health1
EOF

chmod +x "${BIND_WORK}/ray/quick_health_check.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
 bash -lc "echo \$SARGS && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/quick_health_check.sh" "


singularity shell $SARGS "$IMAGE"

# --------------------------- (YOUR TRAINING) -------------------------
# # Replace this stub with your original training command (unchanged flags),
# # just executed under singularity exec on the head:
# pbsdsh -n 0 -- bash -lc "
#   cd /workspace
#   # singularity exec $SARGS '${singularity_image_path}' python -m verl.trainer.cli.train ... 2>&1 | tee verl_demo_pbs.log
#   echo '[PBS] Training stub done (replace with your full command).'
# "

cat > "${BIND_WORK}/ray/run_training.sh" <<EOF
#!/usr/bin/env bash

export RAY_ADDRESS="http://${head_node_ip}:${dashboard_port}"


# Make the project root the working directory Ray ships to workers
export WORKING_DIR="/workspace"

export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

bash recipe/dapo/run_dapo_qwen3_4b_2nodes4A100.sh
EOF

chmod +x "${BIND_WORK}/ray/run_training.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
 bash -lc "echo \$SARGS && echo HELLOT1 && module load singularity && echo HELLOT2 && singularity exec $SARGS "$IMAGE" bash -lc "/workspace/ray/run_training.sh" "



# --------------------------- CLEANUP RAY -----------------------------
cat > "${BIND_WORK}/ray/clean_ray.sh" <<EOF
#!/usr/bin/env bash

ray stop --force || true
EOF

chmod +x "${BIND_WORK}/ray/clean_ray.sh"

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
