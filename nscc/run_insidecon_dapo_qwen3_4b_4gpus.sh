#!/usr/bin/env bash
# set -euo pipefail

# # Hard isolation so ~/.local can’t leak in
# export SINGULARITYENV_PYTHONNOUSERSITE=1
# export SINGULARITYENV_PYTHONPATH=
# export SINGULARITYENV_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

# Start Ray head (inside the container shell/exec)
ray stop --force || true
ray start --head

# Ray dashboard/head address
export RAY_ADDRESS="http://${RAY_IP:-127.0.0.1}:8265"

# Make the project root the working directory Ray ships to workers
export WORKING_DIR="$PWD"
export HOME="/root"

# Point Ray to the runtime env file
export RUNTIME_ENV="$WORKING_DIR/recipe/dapo/runtime_env.yaml"

# (Optional) prove isolation for this parent process
python - <<'PY'
import sys,inspect, transformers
print("exe:", sys.executable)
print("tf_file:", inspect.getfile(transformers))
print("path0:", sys.path[0] if sys.path else "")
PY

# Launch your experiment (keeps current path as cwd for the driver)
bash recipe/dapo/run_dapo_qwen3_4b_4gpus.sh