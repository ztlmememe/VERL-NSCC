#!/bin/bash
set -euo pipefail
cd /users/tzhang/VERL-NSCC/Alps


if [[ " $@ " =~ " --help " ]] || [ -z "$SLURM_JOB_ID" ]; then
    echo "Running a Ray cluster on SLURM with an interactive shell on the head node (to be run in an existing SLURM job allocation)."
    echo ""

    # Can make this an sbatch script by turning these into SBATCH parameters
    # Further requires adding --output instead of redirecting in srun command, removing --pty and interactive bash
    echo "Example: \
salloc\
 --job-name=ray-on-slurm-int\
 --partition=normal\
 --time=01:00:00\
 --nodes=2\
 --ntasks-per-node=1\
 --gpus-per-node=4\
 --cpus-per-task=288\
 ray_on_slurm_interactive.sh
"
    exit 1
fi

ulimit -c 0

# Check that CE_IMAGES points to directory with container images
if [[ -z "${CE_IMAGES}" ]]; then
    echo "Error: CE_IMAGES environment variable is not set." >&2
    exit 1
elif [[ ! -d "${CE_IMAGES}" ]]; then
    echo "Error: The path CE_IMAGES does not exist: ${CE_IMAGES} (should be directory with .sqsh container images)" >&2
    exit 1
fi

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
echo "IP Head: $ip_head"


srun \
    --environment ./env/ngc-ray-25.06.toml --pty \
    bash -c "
        unset no_proxy https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ROCR_VISIBLE_DEVICES

        # activate any virtual env if necessary

        if [[ \"\$(hostname)\" == $head_node ]]; then

            # Register a trap to terminate job step upon exit (alternative to scancel, 'ray stop --force').
            trap 'echo \"Ray head node exiting. Cancelling SLURM step...\" >&2; scancel \$SLURM_JOB_ID.\$SLURM_STEP_ID' EXIT 
            set -e  # exit on error in the head node setup
            
            echo \"Starting head node on \$(hostname)\"
            echo \"Launching Prometheus and Grafana (optional)\"

            ### >>> 运行前做一次 Python/Ray 自检（避免 import ray 失败）
            echo \"[HEAD] Python executable & Ray check:\"
            which python || true
            python -c 'import sys; print(sys.executable)' || true
            python -c 'import sys, platform; print(platform.python_version())' || true
            python -c 'import ray; print(\"ray=\", ray.__version__)'

            set -x

            # Start Ray head node
            # Monitoring dashboard setup, requires 'pip install ray[default]' and use additional ray start --head options:
            # --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
            # Forward port 8265 from the head node to access the dashboard UI.
            ray start --head --node-ip-address=\"$head_node_ip\" --port=$port \
            --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265 \
            --metrics-export-port=8080 \
            --num-cpus \${SLURM_CPUS_PER_TASK} --num-gpus \${SLURM_GPUS_PER_NODE} --block &
            # optional, though may be useful in certain versions of Ray < 1.0.

            # Prometheus setup (optional)
            # https://docs.ray.io/en/latest/cluster/metrics.html#prometheus-setup, use additional ray start --head option:
            # --metrics-export-port=8080
            ray metrics launch-prometheus

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

            bash



        else
            exec > logs/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}-\${SLURM_PROCID}.log 2>&1

            sleep 10

            echo \"Starting worker on \$(hostname) (rank \${SLURM_PROCID})\"
            set -x
            ray start --address \"$ip_head\" --num-cpus \"\${SLURM_CPUS_PER_TASK}\" --num-gpus \"\${SLURM_GPUS_PER_NODE}\" --block

        fi

        echo \"Done with Ray cluster...\"
"
