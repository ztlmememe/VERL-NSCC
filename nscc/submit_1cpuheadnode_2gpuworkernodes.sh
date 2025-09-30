#!/usr/bin/env bash
#PBS -N submit_1cpuheadnode_2gpuworkernodes
#PBS -q normal
#PBS -P 12004167
#PBS -l select=2:ngpus=4
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/submit_1cpuheadnode_2gpuworkernodes.log
#PBS -k oe 
# "-k oe" arg to output log file on the fly, instead of at the end 

# set -xeuo pipefail
cd "$PBS_O_WORKDIR"

export HEAD_NODE_IP="10.168.0.43"

bash nscc/run_2nodes_with_inputheadnodeIP.sh

echo submit_1cpuheadnode_2gpuworkernodes.sh COMPLETED