#!/usr/bin/env bash
#PBS -N deply_ray_head_1node
#PBS -q normal
#PBS -P 12004167
#PBS -l select=1:ngpus=1
#PBS -l walltime=03:30:00
#PBS -j oe
#PBS -o /home/users/ntu/guoweia3/scratch/job_logs/submit_1cpuheadnode.log
#PBS -k oe 
# "-k oe" arg to output log file on the fly, instead of at the end 

# set -xeuo pipefail
cd "$PBS_O_WORKDIR"

bash nscc/run_set_up_ray_head_node.sh

echo submit_1cpuheadnode_wip.sh COMPLETED