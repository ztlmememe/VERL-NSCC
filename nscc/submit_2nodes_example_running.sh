#!/usr/bin/env bash
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
# HEAD_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}' | head -n1)
# HEAD_IP=$(hostname -I | awk '{print $1}' | head -n1)
DEV=$(ip route show default | awk '{print $5; exit}')
HEAD_IP=$(ip -4 addr show "$DEV" | awk '/inet /{sub(/\/.*/,"",$2); print $2; exit}')


GCS_PORT=6379
DASHBOARD_PORT=8265
OBJECT_MANAGER_PORT=8076
NODE_MANAGER_PORT=8077

# # # ---- NCCL (ethernet-friendly defaults; tweak if needed) ----
# export NCCL_SOCKET_IFNAME=$(ip -o link | awk -F': ' '{print $2}' | grep -E 'ib0|hsn|bond0|eno|eth0' | head -n1 || echo eth0)
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=1
# export PYTORCH_NCCL_BLOCKING_WAIT=0
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# # ---- cleanup Ray (both nodes) on exit ----
# cleanup() {
#   for n in "${NODES[@]}"; do
#     pbsdsh -h "$n" bash -lc \
#       "singularity exec $SARGS \"$IMAGE\" bash -lc 'ray stop --force || true'"
#   done
# }
# trap cleanup EXIT





echo starting test code A

# cat > "${RAY_TMP}/start_head_in_container.sh" <<'EOF'
# #!/usr/bin/env bash
# echo hello inside1
# touch testfile2
# ray stop --force || true

# echo gcs_port test 1
# echo ${GCS_PORT}
# echo gcs_port test 2
# echo $GCS_PORT
# echo home test 1
# echo $HOME

# ray start --head \
#   --port=${GCS_PORT} \
#   --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
#   --object-manager-port=${OBJECT_MANAGER_PORT} \
#   --node-manager-port=${NODE_MANAGER_PORT} \
#   --temp-dir=/raytmp \
#   --disable-usage-stats

# echo hello inside1

# EOF


# cat > "${RAY_TMP}/start_head_in_container.sh" <<EOF
# #!/usr/bin/env bash
# echo hello inside1
# touch testfile2
# ray stop --force || true

# echo gcs_port test 1
# echo ${GCS_PORT}
# echo gcs_port test 2
# echo $GCS_PORT
# echo home test 1
# echo $HOME



# ray start --head \
#   --port=${GCS_PORT} \
#   --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
#   --object-manager-port=${OBJECT_MANAGER_PORT} \
#   --node-manager-port=${NODE_MANAGER_PORT} \
#   --temp-dir=/raytmp \
#   --disable-usage-stats

# echo hello inside1
# EOF


cat > "${RAY_TMP}/start_head_in_container.sh" <<EOF
#!/usr/bin/env bash


ray stop --force

echo ray arg1 inside worker ${HEAD_IP}:${GCS_PORT}
echo ray arg2 inside worker ${OBJECT_MANAGER_PORT}
echo ray arg3 inside worker ${NODE_MANAGER_PORT}

ray start --head \
  --port=${GCS_PORT} \
  --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} \
  --object-manager-port=${OBJECT_MANAGER_PORT} \
  --node-manager-port=${NODE_MANAGER_PORT} \
  --temp-dir=/raytmp \
  --disable-usage-stats

EOF


echo starting test code B

chmod +x "${RAY_TMP}/start_head_in_container.sh"

echo starting test code C

# export SINGULARITYENV_GCS_PORT="${GCS_PORT}"
# export SINGULARITYENV_DASHBOARD_PORT="${DASHBOARD_PORT}"
# export SINGULARITYENV_OBJECT_MANAGER_PORT="${OBJECT_MANAGER_PORT}"
# export SINGULARITYENV_NODE_MANAGER_PORT="${NODE_MANAGER_PORT}"


echo starting test code D

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  bash -lc 'echo HELLO1 && module load singularity && echo HELLO2 && singularity shell $SARGS "$IMAGE" && echo HELLO3'
# pbsdsh -n 0 -- env SARGS="${SARGS}" RAY_TMP=$RAY_TMP \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP"

# echo starting test code E

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"   "


singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#     bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"   "

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"   "


# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  bash -lc "module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"   "


# pbsdsh -n 0 -- bash -lc "module load singularity && singularity exec $SARGS $IMAGE bash -lc 'echo hello from container; hostname; nvidia-smi'"

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#  bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2    "


# pbsdsh -n 0 -- bash -lc "hostname -I | awk '{print \$1}'"

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#  module load singularity

# pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" \
#  singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_head_in_container.sh"


echo starting test code F

# ---- start workers (inside containers on remaining nodes) ---- WIP

echo ray head address ${HEAD_IP}:${GCS_PORT}


cat > "${RAY_TMP}/start_worker_in_container.sh" <<EOF
#!/usr/bin/env bash

echo hello inside worker1
touch testfile_from_worker1

hostname -I | awk '{print $1}'

echo ray arg1 inside worker ${HEAD_IP}:${GCS_PORT}
echo ray arg2 inside worker ${OBJECT_MANAGER_PORT}
echo ray arg3 inside worker ${NODE_MANAGER_PORT}

ray start --address=${HEAD_IP}:${GCS_PORT} \
  --object-manager-port=${OBJECT_MANAGER_PORT} \
  --node-manager-port=${NODE_MANAGER_PORT} \
  --temp-dir=/raytmp \
  --disable-usage-stats

echo hello inside worker2
EOF


echo starting test code G

chmod +x "${RAY_TMP}/start_worker_in_container.sh"

# N=$(wc -l < "$PBS_NODEFILE")
# for i in $(seq 1 $((N-1))); do
#   pbsdsh -n "$i" -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
#     bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_worker_in_container.sh"   "
# done

# for i in $(seq 1 $((N-1))); do
#   echo "$i" 
# done

pbsdsh -n 1 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
    bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLOW1 && module load singularity && echo HELLOW2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/start_worker_in_container.sh"   "

echo starting test code H

# ---- quick health check (from head) ---- WIP
cat > "${RAY_TMP}/quick_health_check.sh" <<EOF
#!/usr/bin/env bash

ray status || true
echo hello inside health1
EOF

chmod +x "${RAY_TMP}/quick_health_check.sh"
pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
 bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "/raytmp/quick_health_check.sh" "



echo starting test code I

pbsdsh -n 0 -- env SARGS="${SARGS}" IMAGE="${IMAGE}" RAY_TMP="${RAY_TMP}" \
 bash -lc "echo \$SARGS; echo \$RAY_TMP && echo HELLO1 && module load singularity && echo HELLO2 && singularity exec $SARGS "$IMAGE" bash -lc "export RAY_ADDRESS=http://${HEAD_IP}:${DASHBOARD_PORT}; bash recipe/dapo/run_dapo_qwen3_4b_2nodes_4gpus_1.sh" "



echo starting test code II