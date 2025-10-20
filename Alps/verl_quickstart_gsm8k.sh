#!/bin/bash

set -x

PYTHONUNBUFFERED=1 \
 python3 -m verl.trainer.main_ppo \
 data.train_files=verl-data/gsm8k/train.parquet \
 data.val_files=verl-data/gsm8k/test.parquet \
 data.train_batch_size=$((256 * ${SLURM_NNODES} / 2 )) \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=$((64 * ${SLURM_NNODES} / 2 )) \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node="${SLURM_GPUS_PER_NODE}" \
 trainer.nnodes="${SLURM_NNODES}" \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15