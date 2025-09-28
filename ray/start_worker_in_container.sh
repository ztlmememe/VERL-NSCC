#!/usr/bin/env bash


export WORKING_DIR="/workspace"
export HOME="/root"
# Point Ray to the runtime env file
export RUNTIME_ENV="/workspace/recipe/dapo/runtime_env.yaml"

ray start --address='10.168.0.49:6379'     --num-gpus '4'     --disable-usage-stats 


echo "SLEEPING FOR 10810 s, to keep Ray worker up"
sleep 10810
