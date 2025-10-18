# Recipe: Running Decoupled CLIP and Dynamic Sampling Policy Optimization (DAPO) on NSCC Cluster

This guide explains how to run the **DAPO** project (Decoupled CLIP + Adaptive Policy Optimization) on the NSCC cluster using both **single-node** and **multi-node** Ray training setups.
All commands are assumed to be run inside your **project working directory**.

---

## 🧩 0. Environment and Directory Setup

### Step 1. Load Singularity module

```bash
module load singularity
```

### Step 2. Define Singularity cache and tmp paths

> ⚠️ **Important:** Do *not* let Singularity use your home directory for cache; otherwise, your NSCC 50 GB quota may quickly exceed and lock your account (preventing VSCode login).

```bash
export SINGULARITY_CACHEDIR=/home/users/ntu/<your_id>/scratch/cache/docker_images/.sif_work/
export SINGULARITY_TMPDIR=/home/users/ntu/<your_id>/scratch/cache/docker_images/.sif_work/tmp
export APPTAINER_CACHEDIR=/home/users/ntu/<your_id>/scratch/cache/docker_images/.sif_work/
export APPTAINER_TMPDIR=/home/users/ntu/<your_id>/scratch/cache/docker_images/.sif_work/tmp
```

### Step 3. Pull the public container image

```bash
singularity pull /home/users/ntu/<your_id>/scratch/cache/docker_images/verl_nscc.sif \
    docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
```

---

## 📁 1. Directory Structure

For reproducibility and storage safety, **separate the working and cache directories**.

| Purpose            | Path                                           | Notes                                                     |
| ------------------ | ---------------------------------------------- | --------------------------------------------------------- |
| Working directory  | `/home/users/ntu/<your_id>/scratch/verl`       | All scripts (`nscc/`, `recipe/`, etc.) are run here       |
| Model & data cache | `/home/users/ntu/<your_id>/scratch/cache/verl` | All large files, Hugging Face cache, datasets, and models |

Ensure these folders exist:

```bash
mkdir -p /home/users/ntu/<your_id>/scratch/cache/verl/{data,models}
mkdir -p /home/users/ntu/<your_id>/scratch/cache/verl/models/{hub,datasets,transformers,.hf}
```

---

## 📁 1. Directory Structure

For reproducibility and quota safety, **separate the working directory** (for scripts) **from the cache directory** (for large files).

| Purpose                           | Path                                          | Description                                                                                                    |
| --------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Working directory**             | `/home/users/ntu/tianle00/scratch/verl`       | Contains codebase, `nscc/` submission scripts, and `recipe/` training configs. All commands are executed here. |
| **Cache directory (data/models)** | `/home/users/ntu/tianle00/scratch/cache/verl` | Stores all large files: datasets, checkpoints, Hugging Face caches, and pretrained models.                     |

Recommended to create the following structure (it will be automatically used by all Singularity commands):

```bash
mkdir -p /home/users/ntu/tianle00/scratch/cache/verl/{data,models,ckpts}
mkdir -p /home/users/ntu/tianle00/scratch/cache/verl/models/{hub,datasets,transformers,.hf}
```


---

## 📦 2. Testing Container Mounts and Caches

### Test dataset download (e.g. GSM8K)

```bash
IMAGE=/home/users/ntu/<your_id>/scratch/cache/docker_images/verl_nscc.sif
HOST_CACHE=/home/users/ntu/<your_id>/scratch/cache/verl

singularity exec --cleanenv --no-home \
  --bind "$PWD":/workspace,"$HOST_CACHE/data":/root/verl/data,"$HOST_CACHE/models":/root/verl/models \
  --env HOME=/root,HF_HOME=/root/verl/models/.hf,HF_HUB_CACHE=/root/verl/models/hub,HF_DATASETS_CACHE=/root/verl/models/datasets,TRANSFORMERS_CACHE=/root/verl/models \
  --pwd /workspace "$IMAGE" \
  python3 -m examples.data_preprocess.gsm8k --local_save_dir /root/verl/data/gsm8k
```

### Test model download (interactive Python check)

```bash
singularity exec --cleanenv --no-home \
  --bind "$HOST_CACHE/models":/root/verl/models \
  --env HOME=/root,HF_HOME=/root/verl/models/.hf,HF_HUB_CACHE=/root/verl/models/hub,HF_DATASETS_CACHE=/root/verl/models/datasets,TRANSFORMERS_CACHE=/root/verl/models,HF_HUB_ENABLE_HF_TRANSFER=1 \
  "$IMAGE" bash --noprofile --norc -lc '
python3 - << "PY"
from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Base")
print("ok", type(pipe.model).__name__)
PY
'
```

✅  Successful output should print:

```
ok QwenBlock
```

and the model files will appear under
`/home/users/ntu/<your_id>/scratch/cache/verl/models/models--Qwen--Qwen3-4B-Base`.

---

## 🧮 3. Dataset Preparation

Required datasets:

* **Training:** [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
* **Testing:** [AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024)

Expected locations:

```
/home/users/ntu/<your_id>/scratch/cache/verl/data/dapo-math-17k.parquet
/home/users/ntu/<your_id>/scratch/cache/verl/data/aime-2024.parquet
```

Inside container, these correspond to:

```
/root/verl/data/dapo-math-17k.parquet
/root/verl/data/aime-2024.parquet
```

---

## ⚙️ 4. Model Path Configuration

If you manually downloaded the model, update the path in:

```
/home/users/ntu/<your_id>/scratch/verl/recipe/dapo/run_dapo_qwen3_4b_4gpus_2.sh
```

Example:

```bash
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
```

Do the same for multi-node training script:

```
run_dapo_qwen3_4b_2nodes4A100.sh
```


After running preprocessing and model downloads, your cache folder should look like:

```
/home/users/ntu/tianle00/scratch/cache/verl
│
├── ckpts/
│   └── DAPO/
│       ├── DAPO-Qwen3-4B-Base_4A100/
│       └── DAPO-Qwen3-4B-Base_2nodes4A100_2025-10-16_16:42:30/
│
├── data/
│   ├── gsm8k/
│   ├── aime-2024.parquet
│   └── dapo-math-17k.parquet
│
├── models/
│   ├── .hf/
│   ├── .locks/
│   ├── datasets/
│   ├── hub/
│   ├── transformers/
│   └── models--Qwen--Qwen3-4B-Base/
│       ├── blobs/
│       ├── refs/
│       ├── snapshots/
│       ├── .no_exist
│       └── other auxiliary files
│
└── .triton/
```

**Notes:**

* The `ckpts/DAPO/` directory stores training checkpoints for both **single-node** and **multi-node** runs.
* The `models--Qwen--Qwen3-4B-Base` folder is the Hugging Face auto-downloaded directory structure containing the model weights and snapshot metadata.
* The `.hf`, `.locks`, and `.triton` subdirectories are automatically managed by Hugging Face or Triton runtime; do not delete them unless you need to reset cache.


---

## 🚀 5. Single-Node Training (4 × A100 GPUs)

Submit the training job:

```bash
qsub nscc/submit_dapo_qwen3_4b_4gpus.sh
```

This script launches the Ray cluster **within one node** and starts the training task.

---

## 🧠 6. Multi-Node Training (Ray Cluster)

### Step 1. Launch Ray head node (CPU-only)

```bash
qsub nscc/submit_1cpuheadnode.sh
```

> The Ray **head node** must remain active for the entire training period.
> Set its wall time **longer than** the worker training scripts to prevent job termination.

### Step 2. Check generated head IP

Once the head node job starts, the assigned IP will appear in:

```
ray/start_head_in_container.sh
```

Example line:

```
--node-ip-address='<HEAD_NODE_IP>'
```

⚠️ Make sure the timestamp in that file reflects a *recent* creation (not stale from previous runs):

```
echo "Running start_head_in_container.sh script that was created at <datetime_stamp>"
```

### Step 3. Update worker job script with head IP

Edit:

```bash
export HEAD_NODE_IP="<HEAD_NODE_IP>"
```

in:

```
nscc/submit_2gpuworkernodes_winputheadIP.sh
```

### Step 4. Launch worker nodes

```bash
qsub nscc/submit_2gpuworkernodes_winputheadIP.sh
```

These GPU nodes will automatically join the Ray cluster and receive model training tasks.

---

## 🧹 7. Maintenance and Storage Tips

* Replace all instances of `tianle00` in scripts with your **own NSCC username**.
* Avoid storing large files in `~/` (home directory has only **50 GB** quota).
* Store all large data/models/logs in your `scratch` directory.
* The NSCC system **auto-cleans `scratch/` files not accessed in 3 months** — always back up critical results.

---

## ✅ Quick Summary

| Task                  | Command                                            |
| --------------------- | -------------------------------------------------- |
| Single-node 4 GPU run | `qsub nscc/submit_dapo_qwen3_4b_4gpus.sh`          |
| Multi-node head (CPU) | `qsub nscc/submit_1cpuheadnode.sh`                 |
| Multi-node workers    | `qsub nscc/submit_2gpuworkernodes_winputheadIP.sh` |
