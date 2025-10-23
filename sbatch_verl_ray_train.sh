#!/usr/bin/env bash
#SBATCH -A g204
#SBATCH --job-name=ray-on-slurm-batch
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -euo pipefail
mkdir -p logs
mkdir -p logs/${SLURM_JOB_NAME}
cd /users/tzhang/VERL-NSCC

# --------- Environment Setup (container / venv / proxy etc.) ---------
# If you use containerized environments, you may start enroot/podman here.
# In this case, we continue using the Python virtual environment:
source venv-vllm-v0.10.2/bin/activate || { echo "venv not found"; exit 1; }

# --------- Check image directory (only if using CE_IMAGES container mode) ---------
if [[ -n "${CE_IMAGES:-}" ]]; then
  if [[ ! -d "${CE_IMAGES}" ]]; then
    echo "Warning: CE_IMAGES is set but path not found: ${CE_IMAGES}"
  fi
fi

# --------- Select head node & assign IP ---------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t nodes_array <<<"$nodes"
head_node=${nodes_array[0]}
port=6379

# Obtain head node IP (handle IPv6 / multi-IP cases as in the interactive script)
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPv6 detected; using IPv4: $head_node_ip"
fi
ip_head="${head_node_ip}:${port}"
echo "IP Head: $ip_head"



srun \
    --environment ./Alps/env/ngc-ray-vllm-v0.10.2.toml --pty \
    bash -c "
        unset no_proxy https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ROCR_VISIBLE_DEVICES

        # activate any virtual env if necessary
        source venv-vllm-v0.10.2/bin/activate

        if [[ \"\$(hostname)\" == $head_node ]]; then

            # Register a trap to terminate job step upon exit (alternative to scancel, 'ray stop --force').
            trap 'echo \"Ray head node exiting. Cancelling SLURM step...\" >&2; scancel \$SLURM_JOB_ID.\$SLURM_STEP_ID' EXIT 
            set -e  # exit on error in the head node setup
            
            echo \"Starting head node on \$(hostname)\"
            echo \"Launching Prometheus and Grafana (optional)\"

            set -x

            # Start Ray head node
            # Monitoring dashboard setup, requires 'pip install ray[default]' and use additional ray start --head options:
            # --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
            # Forward port 8265 from the head node to access the dashboard UI.
            python -m ray.scripts.scripts start --head --node-ip-address=\"$head_node_ip\" --port=$port \
            --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265 \
            --metrics-export-port=8080 \
            --num-cpus \${SLURM_CPUS_PER_TASK} --num-gpus \${SLURM_GPUS_PER_NODE} --block &
            # optional, though may be useful in certain versions of Ray < 1.0.

            # Prometheus setup (optional)
            # https://docs.ray.io/en/latest/cluster/metrics.html#prometheus-setup, use additional ray start --head option:
            # --metrics-export-port=8080
            python -m ray.scripts.scripts metrics launch-prometheus

            # Grafana setup (optional)
            # Following https://docs.ray.io/en/latest/cluster/metrics.html#grafana, download Grafana from
            # https://grafana.com/grafana/download?platform=arm and unpack/move to /usr/local/grafana (cf. Dockerfile).
            # Forward port 3000 from the head node to access Grafana web UI.
            set +x
            if ! timeout 30 bash -c 'until [ -f /tmp/ray/session_latest/metrics/grafana/grafana.ini ]; do sleep 1; done'; then
                echo \"Error: Grafana config file not found after 30 seconds.\" >&2
                exit 1
            fi
            set -x
            (cd /usr/local/grafana && bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web) &

            sleep 5

            set +x

            echo \"Waiting all Ray nodes to become alive...\"
            ./ray_on_slurm_wait_for_all_nodes.sh \${SLURM_JOB_NUM_NODES}
            echo \"Testing Ray initialization in the Slurm nodes...\"
            python ray_on_slurm_init_check.py \"$ip_head\"

            set +e  # continue on error in the interactive shell

            echo \"Launch interactive shell...\"  # must be run inside head container

            bash ./verl_quickstart_nscc.sh

            echo \"Training finished. Stopping Ray...\"
            python -m ray.scripts.scripts stop --force || true

        else
            exec > logs/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}-\${SLURM_PROCID}.log 2>&1

            sleep 10

            echo \"Starting worker on \$(hostname) (rank \${SLURM_PROCID})\"
            set -x
            python -m ray.scripts.scripts start --address \"$ip_head\" --num-cpus \"\${SLURM_CPUS_PER_TASK}\" --num-gpus \"\${SLURM_GPUS_PER_NODE}\" --block

        fi

        echo \"Done with Ray cluster...\"
"


exit ${TRAIN_RC:-0}
