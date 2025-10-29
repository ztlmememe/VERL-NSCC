#!/usr/bin/env bash
#PBS -N submit_1cpuheadnode_2gpuworkernodes
#PBS -q normal
#PBS -P 12004167
#PBS -l select=2:ngpus=4
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/users/ntu/tianle00/scratch/verl/job_logs/submit_2gpuworkernodes_winputheadIP.log

# "-k oe" arg to output log file on the fly, instead of at the end 

# set -xeuo pipefail
cd "$PBS_O_WORKDIR"

# to replace HEAD_NODE_IP with the actual HEAD_NODE_IP from `ray/start_head_in_container.sh`. 
# Important: check the datetime stamp in `ray/start_head_in_container.sh` script to make sure HEAD_NODE_IP is not stale (from previous runs) in the line `echo "Running start_head_in_container.sh script that was created at <datetime_stamp>"`
export HEAD_NODE_IP='10.168.4.101'


bash nscc/run_2nodes_with_inputheadnodeIP.sh

echo submit_2gpuworkernodes_winputheadIP.sh COMPLETED