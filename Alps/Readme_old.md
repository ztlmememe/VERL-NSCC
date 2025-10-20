# Running a Ray cluster on Alps

A set of utilities to run a [Ray](https://github.com/ray-project/ray) cluster with an interactive shell on Alps through SLURM. This allows to run individual Ray commands as well as more complex Ray applications like Reinforcement Learning with [verl](https://github.com/volcengine/verl) interactively from the head node.

## How to launch a Ray cluster with an interactive shell

The main script to launch an interactive session is `ray_on_slurm_interactive.sh`. The following describes an example how to launch a Ray cluster (cf. its `--help` message).

First, clone this repository in a path under `${SCRATCH}`. In order to run a 2-node Ray cluster and launch an interactive shell on the Ray head node run the following command on the Clariden login node `clariden-lnXXX`:

```bash
[user@clariden-lnXXX slurm-ray-demo]$ salloc --job-name=ray-on-slurm-int\
 --partition=normal\
 --time=01:00:00\
 --nodes=2\
 --ntasks-per-node=1\
 --gpus-per-node=4\
 --cpus-per-task=288\
 ray_on_slurm_interactive.sh
```

This expects the environment variable `CE_IMAGES` to point to the directory with your container images in `.sqsh` files (so that the `image` entry in the environment definition files under `env/*.toml` resolve to a valid path). See below for how to build the container image.

The above command allocates a SLURM job and runs the script `ray_on_slurm_interactive.sh`. The output you will see comes from rank 0 of the SLURM job, that will be launching in this order a Ray head node, Prometheus and a Grafana instance. Afterwards, a check is performed that all workers joined the Ray cluster and an interactive shell is opened. If this runs successfully, it should look like this:

```
=== Ray Cluster Status ===
Number of nodes: 2
Node: nid00YYYY, Status: True
Node: nid00ZZZZ, Status: True
Ray initialization successful!
Launch interactive shell...
user@nid00YYYY:/path/to/slurm-ray-demo$ ... # run your Ray commands here
```

The compute node `nid00YYYY` is where the Ray head is running. Now, you can execute commands one-by-one such as in the [Ray documentation](https://docs.ray.io/en/latest/ray-core/examples/gentle_walkthrough.html) in order to familiarize with Ray concepts interactively and test workflows. In order to monitor the cluster the Ray dashboard and Grafana can be made available locally - for this, set up port-forwarding to your local machine and access the corresponding websites at [http://localhost:8265]() and [http://localhost:3000](), respectively.

Once you are done with the Ray cluster, you can exit the interactive shell, which will cancel the job step.

### Building the container image

The corresponding container image is based on a [NGC PyTorch release](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and can be built (on a compute node) with

```bash
[user@nid00AAAA slurm-ray-demo]:/path/to/slurm-ray-demo$ cd env
[user@nid00AAAA env]$ podman build -f Dockerfile -t ${USER}/ngc-ray:25.06 .
[user@nid00AAAA env]$ enroot import -x mount -o ${CE_IMAGES}/ngc-ray+25.06.sqsh podman://${USER}/ngc-ray:25.06 
```

where the variable `${CE_IMAGES}` points to the directory with container images.

## Running Reinforcement Learning with verl on Alps

To run Reinforcement Learning with [verl](https://github.com/volcengine/verl) and [vLLM](https://github.com/vllm-project/vllm) on the [GSM8k quickstart example](https://verl.readthedocs.io/en/latest/start/quickstart.html), first download the dataset (to `verl-data`) and model following the instructions in verl's documentation.

### Building the container image and virtual environment on top

The container image is based on the [vLLM base image](https://hub.docker.com/r/vllm/vllm-openai/) and can be built (on a compute node) with (`${CE_IMAGES}` pointing to the directory with container images)

```bash
[user@nid00AAAA slurm-ray-demo]$ cd env
[user@nid00AAAA env]$ podman build -f Dockerfile.ray-vllm -t ${USER}/ngc-ray-vllm:v0.10.2 .
[user@nid00AAAA env]$ enroot import -x mount -o ${CE_IMAGES}/ngc-ray-vllm+v0.10.2.sqsh podman://${USER}/ngc-ray-vllm:v0.10.2 
```

This may take a while due to building the flash-attention package.

Then, set up a virtual environment on top of this image and install verl in editable mode by running an interactive shell inside the container as follows:

```bash
[user@clariden-lnXXX slurm-ray-demo]$ git clone git@github.com:volcengine/verl.git
[user@clariden-lnXXX slurm-ray-demo]$ srun --environment ./env/ngc-ray-vllm-v0.10.2.toml --pty bash
user@nid00AAAA:/path/to/slurm-ray-demo$ python3 -m venv --system-site-packages venv-vllm-v0.10.2 && \
  source venv-vllm-v0.10.2/bin/activate && \
  pip install --no-build-isolation -e ./verl
```

This allows to make changes to the source tree cloned at `verl` and test them quickly, which is helpful for development. Once, the codebase has stabilized, verl can also be installed as part of the Dockerfile (so that no virtual environment is needed on top anymore).

### Running the verl quickstart example

Once the container image and virtual environment are set up, launch a Ray cluster with `verl_ray_on_slurm_interactive.sh` (uses `python -m ray.scripts.scripts` in place of `ray` to use the activated virtual environment in Ray workers).

Then, once the interactive shell on the Ray head node has started, run `verl_quickstart_gsm8k.sh`. As above, this expects the environment variable `CE_IMAGES` to point to the directory with your container images in `.sqsh` files (so that the `image` entry in the environment definition files under `env/*.toml` resolve to a valid path).