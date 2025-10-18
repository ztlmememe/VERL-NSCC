# Recipe: Running Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) on NSCC Cluster with Single and Multi-node Training


## Single-Node Example

1. Submit job to set up Ray cluster and assigns model training tasks to Ray cluster:
```bash
qsub nscc/submit_dapo_qwen3_4b_4gpus.sh
```


## Multiple-Node Example

1. Submit job to set up Ray cluster head (CPU-only node) which assigns model training tasks to Ray cluster workers:

```bash
qsub nscc/submit_1cpuheadnode.sh
```

Ray cluster head has to be set up as a separate node from the worker nodes to avoid the head from OOM during periods of high memory load from model training.


After submitting the Ray head's IP will be written in `ray/start_head_in_container.sh` like `--node-ip-address='<HEAD_NODE_IP>'`. This might take a while depending on how soon NSCC assign nodes to run the `nscc/submit_1cpuheadnode.sh` job.  
Important: check the datetime stamp in `ray/start_head_in_container.sh` script to make sure HEAD_NODE_IP is not stale (from previous runs) in the line `echo "Running start_head_in_container.sh script that was created at <datetime_stamp>"`.


2. Edit to insert the `'<HEAD_NODE_IP>'` in `nscc/submit_2gpuworkernodes_winputheadIP.sh` so that the Ray workers can find the Ray head to receive model training task at the next step:

```bash
export HEAD_NODE_IP="<HEAD_NODE_IP>"
```


3. Submit job to set up Ray cluster workers (GPU nodes) which take model training tasks from Ray cluster head:

```bash
qsub nscc/submit_2gpuworkernodes_winputheadIP.sh
```




