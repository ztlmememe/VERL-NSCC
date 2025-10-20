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


---

## 🧩 1. Configure Podman Storage

Before using `podman build`, you must configure a local storage directory that can handle temporary container layers.
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
# Option A: One-step allocation and login
srun --partition=normal --time=02:00:00 --nodes=1 -A g204 \
     --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=0 \
     --pty bash

# Option B: Two-step allocation
salloc --partition=normal --time=02:00:00 --nodes=1 -A g204 \
       --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=0
srun --jobid=<JOBID> --pty bash
```

Then build the base and vLLM images:

```bash
podman build -f Dockerfile -t ${USER}/ngc-ray:25.06 .
podman build -f Dockerfile.ray-vllm -t ${USER}/ngc-ray-vllm:v0.10.2 .
```

> Building `flash-attention` during this step can take several minutes.

---

## 🗂️ 3. Export the Image to Persistent Storage

Both the **build** and **export** must happen in the **same job** (same Podman session),
because temporary storage in `/dev/shm` is deleted after the job ends.

```bash
export CE_IMAGES=/capstor/scratch/cscs/$USER/images
mkdir -p $CE_IMAGES
```

Then export the built Podman image as a `.sqsh` file:

```bash
enroot import -x mount -o ${CE_IMAGES}/ngc-ray+25.06.sqsh podman://localhost/${USER}/ngc-ray:25.06
enroot import -x mount -o ${CE_IMAGES}/ngc-ray-vllm+v0.10.2.sqsh podman://localhost/${USER}/ngc-ray-vllm:v0.10.2
```

> ✅ If you see `[INFO] Fetching image`, it means Enroot did not find your local image —
> use `podman images` to confirm the repository name and replace it with the `localhost/...` prefix.

---

## ⚙️ 4. Update Environment Definition Files

Open `/users/<username>/VERL-NSCC/Alps/env/ngc-ray-25.06.toml`
and update the following:

1. **`image`** → set to your exported `.sqsh` file path:

   ```toml
   image = "/capstor/scratch/cscs/<username>/images/ngc-ray+25.06.sqsh"
   ```
2. **`mounts`** → add your user directory:

   ```toml
   mounts = [
       "/capstor",
       "/iopsstor",
       "/users/${USER}"
   ]
   ```

---

## ⚠️ 5. Disable Conda Auto-Activation

If Conda auto-starts in your shell (for example, it modifies `$PATH` to point to your local Miniconda),
it will **override the Python inside the container**, breaking the Ray environment.

To disable Conda auto-activation:

```bash
conda config --set auto_activate_base false
```

Or comment out the “conda initialize” block in `~/.bashrc`.

---

## 🧍 6. Fix “I have no name!” Issue (Optional)

When entering the container, you might see:

```
groups: cannot find name for group ID 33318
I have no name!@nid00XXXX:~$
```

Add the following to the top of your `~/.bashrc` to provide a safe prompt and skip the group lookup:

```bash
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  export PS1="[${USER:-u$(id -u)}@$(hostname -s) \W]\\$ "
  return 0
fi
```

This does **not** affect execution or Ray functionality.

---

## 🧠 7. Test Ray-on-Slurm Cluster Launch

Run the Ray interactive launch script to verify the head/worker allocation:

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

When everything runs correctly, you’ll see Ray cluster logs followed by training or evaluation progress similar to:

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

Start a container session and create a temporary virtual environment:

```bash
srun -A g204 --environment ./env/ngc-ray-vllm-v0.10.2.toml --pty bash
cd /users/tzhang/VERL-NSCC

python3 -m venv --system-site-packages venv-vllm-v0.10.2 && \
  source venv-vllm-v0.10.2/bin/activate && \
  pip install --no-build-isolation -e .
```

Next time, simply reactivate:

```bash
source venv-vllm-v0.10.2/bin/activate
```

---

## 🖥️ 9. Launch the Ray Head Node

Exit any previous container sessions:

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

You should eventually see:

```
=== Ray Cluster Status ===
Number of nodes: 2
Node: nid007203, Status: True
Node: nid007174, Status: True
Ray initialization successful!
Launch interactive shell...
```

---

## 🔍 10. Run GSM8K Example Inside the Head Node

Once the interactive shell launches, re-activate your virtual environment:

```bash
source venv-vllm-v0.10.2/bin/activate
```

Then download the example dataset and model:

```bash
python3 -m examples.data_preprocess.gsm8k --local_save_dir /users/tzhang/VERL-NSCC/verl-data/gsm8k
```

Finally, run the VERL quickstart:

```bash
bash ./verl_quickstart_gsm8k.sh
```


---

## 🧹 11. Cleanup

After you finish:

```bash
exit       # Leave Ray head interactive shell
scancel <job_id>  # If any worker jobs are still running
```

---

