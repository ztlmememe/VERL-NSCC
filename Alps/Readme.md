# 🚀 Running VERL + Ray on SLURM (CSCS ALPS Environment)

This guide explains how to build and run the VERL reinforcement learning environment based on the **vLLM** container image within the **CSCS ALPS** system using **Podman**, **Enroot**, and **Slurm**.

---

## 📘 Summary Workflow

| Step | Action                        | Command                             |
| ---- | ----------------------------- | ----------------------------------- |
| 1    | Configure Podman storage      | `~/.config/containers/storage.conf` |
| 2    | Build image on compute node   | `podman build -f Dockerfile...`     |
| 3    | Export to `.sqsh`             | `enroot import ...`                 |
| 4    | Update TOML paths             | edit `env/ngc-ray-*.toml`           |
| 5    | Disable Conda auto-activation | edit `~/.bashrc`                    |
| 6    | Fix user prompt (optional)    | add `getent` block                  |
| 7    | Test Ray cluster launch       | `ray_on_slurm_interactive.sh`       |
| 8    | Create venv + install VERL    | `python3 -m venv ...`               |
| 9    | Launch head node              | `verl_ray_on_slurm_interactive.sh`  |
| 10   | Run quickstart                | `bash verl_quickstart_gsm8k.sh`     |
| 11   | Run NSCC-compatible pipeline  | `bash verl_quickstart_nscc.sh`      |

---

## 🧩 1. Configure Podman Storage

Before using `podman build`, configure local storage that can handle temporary container layers.

Create the file:
`/users/<username>/.config/containers/storage.conf`

```ini
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
```

> **Why:**
> `/users/...` is not writable by rootless Podman on CSCS; therefore, Podman must use a shared temporary storage area under `/dev/shm`.
> These layers are ephemeral — you will later export them into a persistent `.sqsh` file.

---

## 🧮 2. Build the Container Image on a Compute Node

Obtain a compute node allocation (2 hours is usually enough):

```bash
srun --partition=normal --time=02:00:00 --nodes=1 -A g204 \
     --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=0 \
     --pty bash
```

Then build the base and vLLM images:

```bash
podman build -f Dockerfile -t ${USER}/ngc-ray:25.06 .
podman build -f Dockerfile.ray-vllm -t ${USER}/ngc-ray-vllm:v0.10.2 .
```

> Building `flash-attention` during this step can take several minutes.

---

## 🗂️ 3. Export the Image to Persistent Storage

Both **build** and **export** must happen in the same Podman session (same job),
because temporary storage in `/dev/shm` is deleted after the job ends.

```bash
export CE_IMAGES=/capstor/scratch/cscs/$USER/images
mkdir -p $CE_IMAGES
```

Then export the image:

```bash
enroot import -x mount -o ${CE_IMAGES}/ngc-ray+25.06.sqsh podman://localhost/${USER}/ngc-ray:25.06
enroot import -x mount -o ${CE_IMAGES}/ngc-ray-vllm+v0.10.2.sqsh podman://localhost/${USER}/ngc-ray-vllm:v0.10.2
```

> ✅ If you see `[INFO] Fetching image`, it means Enroot did not find your local image — use `podman images` to confirm and include the `localhost/...` prefix.

---

## ⚙️ 4. Update Environment Definition Files

Edit `/users/<username>/VERL-NSCC/Alps/env/ngc-ray-25.06.toml`:

```toml
image = "/capstor/scratch/cscs/<username>/images/ngc-ray+25.06.sqsh"
mounts = [
    "/capstor",
    "/iopsstor",
    "/users/${USER}"
]
```

> ⚠️ **Important Update:**
> Currently, all large image files are stored under `/capstor/scratch/cscs`.
> After the **November 10 system update**, a new scratch directory `/ritom/scratch/cscs` will become available.
> When that happens, you must update all `image` paths and **re-export or rebuild** the container images to the new location.

---

## ⚠️ 5. Disable Conda Auto-Activation

Disable Conda’s base environment auto-start:

```bash
conda config --set auto_activate_base false
```

Or comment out the initialization block in `~/.bashrc`.

---

## 🧍 6. Fix “I have no name!” Issue (Optional)

If you see:

```
groups: cannot find name for group ID 33318
I have no name!@nid00XXXX:~$
```

Add to `~/.bashrc`:

```bash
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  export PS1="[${USER:-u$(id -u)}@$(hostname -s) \W]\\$ "
  return 0
fi
```

---

## 🧠 7. Test Ray-on-Slurm Cluster Launch

To test a small 2-node interactive Ray cluster:

```bash
salloc -A g204 --job-name=ray-on-slurm-int \
 --partition=normal \
 --time=01:00:00 \
 --nodes=2 \
 --ntasks-per-node=1 \
 --gpus-per-node=4 \
 --cpus-per-task=288 \
 ray_on_slurm_interactive.sh
```

Expected output:

```
=== Ray Cluster Status ===
Number of nodes: 2
Node: nid007203, Status: True
Node: nid007174, Status: True
Ray initialization successful!
...
```

---

## 🧩 8. Launch RL Test Environment

Start an interactive session and set up your environment:

```bash
srun -A g204 --environment ./env/ngc-ray-vllm-v0.10.2.toml --pty bash
cd /users/tzhang/VERL-NSCC

python3 -m venv --system-site-packages venv-vllm-v0.10.2
source venv-vllm-v0.10.2/bin/activate
pip install --no-build-isolation -e .
```

Re-activate later with:

```bash
source venv-vllm-v0.10.2/bin/activate
```

---

## 🖥️ 9. Launch the Ray Head Node

```bash
cd /users/tzhang/VERL-NSCC

salloc -A g204 --job-name=ray-on-slurm-int \
 --partition=normal \
 --time=01:00:00 \
 --nodes=2 \
 --ntasks-per-node=1 \
 --gpus-per-node=4 \
 --cpus-per-task=288 \
 verl_ray_on_slurm_interactive.sh
```

---

## 🔍 10. Run GSM8K Example Inside the Head Node

After entering the interactive shell:

```bash
source venv-vllm-v0.10.2/bin/activate
python3 -m examples.data_preprocess.gsm8k --local_save_dir /users/tzhang/VERL-NSCC/verl-data/gsm8k
bash ./verl_quickstart_gsm8k.sh
```

---

## 🌏 11. NSCC-Compatible Workflow (Comparison & Notes)

For users migrating from the **NSCC** environment:

| Environment   | Image Architecture | Container Runtime | Notes                                             |
| ------------- | ------------------ | ----------------- | ------------------------------------------------- |
| **NSCC**      | `linux/amd64`      | Singularity       | Default x86 images work directly                  |
| **CSCS ALPS** | `linux/arm64`      | Podman + Enroot   | Must rebuild locally due to architecture mismatch |

👉 **Why rebuild:**
Most public images are `amd64` only. CSCS ALPS uses `arm64` nodes, so you **must** build from Dockerfiles using the configuration provided in this guide.
However, once rebuilt, the environment supports running all previous NSCC VERL experiments seamlessly.

### ✅ Interactive Test (same as NSCC)

```bash
cd /users/tzhang/VERL-NSCC/

salloc -A g204 --job-name=ray-on-slurm-int \
 --partition=normal \
 --time=01:00:00 \
 --nodes=2 \
 --ntasks-per-node=1 \
 --gpus-per-node=4 \
 --cpus-per-task=288 \
 verl_ray_on_slurm_interactive.sh

source venv-vllm-v0.10.2/bin/activate
bash ./verl_quickstart_nscc.sh  # Equivalent to ./recipe/dapo/run_dapo_qwen3_4b_2nodes4A100.sh
```

### 🧾 Non-Interactive (Batch Mode)

If you don’t want an interactive session, submit via Slurm:

```bash
sbatch ./sbatch_verl_ray_train.sh
```


> ⚠️ **Note:**
> Although `verl_quickstart_nscc.sh` uses the **same parameters** as
> `./recipe/dapo/run_dapo_qwen3_4b_2nodes4A100.sh`,
> it has been **slightly modified** to better match the CSCS environment.
> When testing **other training or evaluation scripts**, you should similarly adjust these configurations to fit your environment.
>
> The same principle applies to `sbatch_verl_ray_train.sh`:
> when submitting non-interactive jobs, edit the line inside the script that runs
>
> ```bash
> srun ... bash ./verl_quickstart_nscc.sh
> ```
>
> and replace it with the specific script you intend to run.
> All other job configuration parameters (partition, GPUs, CPUs, etc.) can remain unchanged.


> ⚠️ **Checkpoint Warning:**
> Each checkpoint file can be ~47 GB.
> Do **not** store checkpoints under `/users/...` (home quota is limited).
> Instead, modify the script to save under `/capstor/scratch/cscs/$USER/` or, after the November 10 update, under `/ritom/scratch/cscs/$USER/`.


---

## 🧹 12. Cleanup

```bash
exit             # Leave Ray interactive shell
scancel <job_id> # Stop worker jobs if still running
```
