#!/usr/bin/env bash
#PBS -N dapo_ray_2n4g_sing
#PBS -q normal
#PBS -P 12004167
#PBS -l select=2:ngpus=4
#PBS -l walltime=03:00:00
#PBS -l place=scatter
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/dapo_ray_2n4g_sing.log

# set -xeuo pipefail
cd "$PBS_O_WORKDIR"

module load singularity

# ---- container & binds (same style as your single-node) ----
IMAGE="/home/users/ntu/guoweia3/scratch/images/verl_nscc.sif"
BIND_WORK="$PWD"                                    # project code
BIND_DATA="/home/users/ntu/guoweia3/scratch/experiments/data/agents"  # shared data
RAY_TMP="/home/users/ntu/guoweia3/scratch/ray-${PBS_JOBID}"           # per-job ray temp
mkdir -p "$RAY_TMP"

# Common Singularity args
SARGS="--nv --cleanenv --no-home --env HOME=/root \
  --bind ${BIND_WORK}:/workspace,${BIND_DATA}:/root,${RAY_TMP}:/raytmp \
  --pwd /workspace"

# ---- nodes & ports ----
mapfile -t NODES < <(uniq "${PBS_NODEFILE}")
HEAD_NODE="${NODES[0]}"
HEAD_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}' | head -n1)

GCS_PORT=6379
DASHBOARD_PORT=8265
OBJECT_MANAGER_PORT=8076
NODE_MANAGER_PORT=8077

# # ---- NCCL (ethernet-friendly defaults; tweak if needed) ----
# export NCCL_SOCKET_IFNAME=$(ip -o link | awk -F': ' '{print $2}' | grep -E 'ib0|hsn|bond0|eno|eth0' | head -n1 || echo eth0)
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=1
# export PYTORCH_NCCL_BLOCKING_WAIT=0
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ---- cleanup Ray (both nodes) on exit ----
cleanup() {
  N=$(wc -l < "$PBS_NODEFILE")
  for i in $(seq 0 $((N-1))); do
    pbsdsh -h "$n" bash -lc \
      "singularity exec $SARGS \"$IMAGE\" bash -lc 'ray stop --force || true'"
  done
}
trap cleanup EXIT



# ---- start head (inside container on head node) ----
cat > "${RAY_TMP}/start_head_in_container.sh" <<EOF
#!/usr/bin/env bash
echo hello inside1
touch testfile2
ray stop --force || true

echo gcs_port test 1
echo ${GCS_PORT}
echo gcs_port test 2
echo $GCS_PORT
echo home test 1
echo $HOME

ray start --head \
  --port=${GCS_PORT} \
  --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
  --object-manager-port=${OBJECT_MANAGER_PORT} \
  --node-manager-port=${NODE_MANAGER_PORT} \
  --temp-dir=/raytmp \
  --disable-usage-stats

echo hello inside1
EOF

chmod +x "${RAY_TMP}/start_head_in_container.sh"

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
 bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"   "

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"  && echo HELLO3 && echo \$SARGS; echo \$RAY_TMP"



# ---- start workers (inside containers on remaining nodes) ---- WIP
cat > "${RAY_TMP}/start_worker_in_container.sh" <<EOF
#!/usr/bin/env bash
echo hello inside worker1
touch testfile_from_worker1
ray stop --force || true

ray start --address=${HEAD_IP}:${GCS_PORT} \
  --object-manager-port=${OBJECT_MANAGER_PORT} \
  --node-manager-port=${NODE_MANAGER_PORT} \
  --temp-dir=/raytmp \
  --disable-usage-stats

echo hello inside worker2
EOF

chmod +x "${RAY_TMP}/start_worker_in_container.sh"

N=$(wc -l < "$PBS_NODEFILE")
for i in $(seq 1 $((N-1))); do
  pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
    bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_worker_in_container.sh"   "
done


# ---- quick health check (from head) ---- WIP
cat > "${RAY_TMP}/quick_health_check.sh" <<EOF
#!/usr/bin/env bash

ray status || true
echo hello inside health1
EOF

chmod +x "${RAY_TMP}/quick_health_check.sh"
pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
 bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/quick_health_check.sh" "












# ---- start workers (inside containers on remaining nodes) ----
N=$(wc -l < "$PBS_NODEFILE")
for i in $(seq 1 $((N-1))); do
  pbsdsh -n "$i" -- bash -lc "
    module load singularity
    singularity exec $SARGS \"$IMAGE\" bash -lc '
      ray stop --force || true;
      ray start --address=${HEAD_IP}:${GCS_PORT} \
        --object-manager-port=${OBJECT_MANAGER_PORT} \
        --node-manager-port=${NODE_MANAGER_PORT} \
        --temp-dir=/raytmp \
        --disable-usage-stats
    '
  "
done

# ---- quick health check (from head) ----
pbsdsh -n 0 -- bash -lc "
  singularity exec $SARGS \"$IMAGE\" bash -lc 'ray status || true
  '
"



# # ---- start head (inside container on head node) ----
# pbsdsh -n 0 -- bash -lc "
#   module load singularity
#   singularity exec $SARGS \"$IMAGE\" bash -lc '
#     ray stop --force || true;
#     ray start --head \
#       --port=${GCS_PORT} \
#       --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
#       --object-manager-port=${OBJECT_MANAGER_PORT} \
#       --node-manager-port=${NODE_MANAGER_PORT} \
#       --temp-dir=/raytmp \
#       --disable-usage-stats
#   '
# "


# pbsdsh -n 0 -- bash -lc "bash ${RAY_TMP}/remote.sh"

# pbsdsh -n 0 -- env SARGS="${SARGS}" RAY_TMP=$RAY_TMP \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP"


# pbsdsh -n 0 -- env SARGS="${SARGS}" RAY_TMP=$RAY_TMP \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP; module load singularity; echo hello "


# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE=$IMAGE  RAY_TMP=$RAY_TMP \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP; module load singularity; singularity shell $SARGS \"$IMAGE\""


# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE=$IMAGE  RAY_TMP=$RAY_TMP \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP; module load singularity; singularity exec $SARGS \"$IMAGE\" bash -lc 'echo inside singularity container' "


# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE=${IMAGE}  RAY_TMP=${RAY_TMP} \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP; module load singularity; singularity exec $SARGS \"$IMAGE\" bash -lc 'touch testfile' "

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE=${IMAGE}  RAY_TMP=${RAY_TMP} \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP; module load singularity; singularity exec $SARGS \"$IMAGE\" bash -lc 'ray stop --force || true; ray start --head --port=${GCS_PORT} --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} --object-manager-port=${OBJECT_MANAGER_PORT} --node-manager-port=${NODE_MANAGER_PORT} --temp-dir=/raytmp --disable-usage-stats' "



# cat > "${RAY_TMP}/start_head_in_container.sh" <<'EOF'
# #!/usr/bin/env bash
# echo hello
# touch testfile1
# ray stop --force || true
# ray start --head \
#   --port=${GCS_PORT} \
#   --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
#   --object-manager-port=${OBJECT_MANAGER_PORT} \
#   --node-manager-port=${NODE_MANAGER_PORT} \
#   --temp-dir=/raytmp \
#   --disable-usage-stats
# EOF

# chmod +x "${RAY_TMP}/start_head_in_container.sh"

# export SINGULARITYENV_GCS_PORT="${GCS_PORT}"
# export SINGULARITYENV_DASHBOARD_PORT="${DASHBOARD_PORT}"
# export SINGULARITYENV_OBJECT_MANAGER_PORT="${OBJECT_MANAGER_PORT}"
# export SINGULARITYENV_NODE_MANAGER_PORT="${NODE_MANAGER_PORT}"


# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  bash -lc 'module load singularity ; singularity exec $SARGS "$IMAGE" bash -lc "bash /raytmp/start_head_in_container.sh"'





# ---- start workers (inside containers on remaining nodes) ----
N=$(wc -l < "$PBS_NODEFILE")
for i in $(seq 1 $((N-1))); do
  pbsdsh -n "$i" -- bash -lc "
    module load singularity
    singularity exec $SARGS \"$IMAGE\" bash -lc '
      ray stop --force || true;
      ray start --address=${HEAD_IP}:${GCS_PORT} \
        --object-manager-port=${OBJECT_MANAGER_PORT} \
        --node-manager-port=${NODE_MANAGER_PORT} \
        --temp-dir=/raytmp \
        --disable-usage-stats
    '
  "
done

# ---- quick health check (from head) ----
pbsdsh -n 0 -- bash -lc "
  singularity exec $SARGS \"$IMAGE\" bash -lc 'ray status || true
  '
"





pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
 bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "export RAY_ADDRESS=http://${HEAD_IP}:${DASHBOARD_PORT}; bash recipe/dapo/run_dapo_qwen3_4b_2nodes_4gpus_1.sh" "



# ---- submit your job INSIDE the head container ----
# Place your submit script as: /workspace/submit_dapo_job.sh
pbsdsh -n 0 -- bash -lc "
  export RAY_ADDRESS=http://${HEAD_IP}:${DASHBOARD_PORT};
   singularity exec $SARGS \"$IMAGE\" bash -lc '
     export RAY_ADDRESS=http://${HEAD_IP}:${DASHBOARD_PORT};
     bash recipe/dapo/run_dapo_qwen3_4b_2nodes_4gpus_1.sh
   '"

wait
