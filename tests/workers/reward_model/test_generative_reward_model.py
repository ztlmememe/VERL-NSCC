# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import ray
import torch

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import HFModelConfig, RewardModelConfig, RewardModelDataProcessorConfig, SamplingConfig
from verl.workers.roles import RewardModelWorker


def create_data_samples(tokenizer) -> DataProto:
    convs = [
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between -1 and 1."},
        ],
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between 0 and 1."},
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Canberra is the capital city of Australia.",
            },
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Sydney is the capital of Australia.",
            },
        ],
    ]

    prompt_length, response_length = 1024, 4096
    pad_token_id = tokenizer.pad_token_id
    prompts, responses, input_ids, attention_masks = [], [], [], []
    for conv in convs:
        prompt_tokens = tokenizer.apply_chat_template(conv[:1], tokenize=True)
        response_tokens = tokenizer.apply_chat_template(conv, tokenize=True)[len(prompt_tokens) :]

        padded_prompt = [pad_token_id] * (prompt_length - len(prompt_tokens)) + prompt_tokens
        padded_response = response_tokens + [pad_token_id] * (response_length - len(response_tokens))
        attention_mask = (
            [0] * (prompt_length - len(prompt_tokens))
            + [1] * len(prompt_tokens)
            + [1] * len(response_tokens)
            + [0] * (response_length - len(response_tokens))
        )
        prompts.append(torch.tensor(padded_prompt))
        responses.append(torch.tensor(padded_response))
        input_ids.append(torch.tensor(padded_prompt + padded_response))
        attention_masks.append(torch.tensor(attention_mask))

    prompts = torch.stack(prompts)
    responses = torch.stack(responses)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    position_ids = compute_position_id_with_mask(attention_masks)

    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
        },
    )


def test_reward_model():
    ray.init()

    rm_path = os.path.expanduser("~/models/verl-team/GenRM-CI-Test-1.5B")
    model_config = HFModelConfig(path=rm_path)
    sampling_config = SamplingConfig(temperature=0.0)
    data_processor_config = RewardModelDataProcessorConfig(
        path="tests/workers/reward_model/process_fn.py",
        preprocess_fn_name="construct_genrm_inputs_from_rollouts",
        postprocess_fn_name="convert_genrm_responses_to_rewards",
    )
    config = RewardModelConfig(
        enable=True,
        model_type="generative",
        max_new_tokens=4096,
        dtype="bfloat16",
        model_config=model_config,
        input_model_config=None,
        tensor_model_parallel_size=2,
        sampling_config=sampling_config,
        data_processor_config=data_processor_config,
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(RewardModelWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[8])
    rm_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    rm_wg.init_model()

    # create data samples
    tokenizer = model_config.get_processor()
    data = create_data_samples(tokenizer)

    gen_batch = rm_wg.compute_rm_score(data)
    server_rm_scores = gen_batch.batch["rm_scores"].sum(dim=-1)
    print(f"{server_rm_scores=}")

    ray.shutdown()
