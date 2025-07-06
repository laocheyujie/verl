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

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import aiohttp
import torch

from verl import DataProto
from verl.workers.reward_manager.registry import register


@dataclass
class ExternalRewardScoreInput:
    data_source: str
    prompt_str: str
    response_str: str
    sequence_str: str
    ground_truth: str
    extra_info: str
    valid_response_length: int

@dataclass
class ScoreFuncOutput:
    score: float
    extra_info: dict


async def async_call_online_reward_model(url: str, method: str = "POST", **kwargs):
    try:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            if method == "POST":
                async with session.post(url, headers=headers, json=kwargs) as response:
                    res = await response.json()
                    score = res.get("score")
                    extra_info = res.get("extra_info", {})
                    return float(score), extra_info
            elif method == "GET":
                async with session.get(url, headers=headers, params=kwargs) as response:
                    res = await response.json()
                    score = res.get("score")
                    extra_info = res.get("extra_info", {})
                    return float(score), extra_info
            else:
                raise ValueError(f"Invalid reward api method: {method}")
    except Exception:
        return 0.0, {}


def call_online_reward_model(url: str, method: str = "POST", **kwargs):
    try:
        # Use the async function in a synchronous context
        loop = asyncio.get_event_loop()
        score, extra_infos = loop.run_until_complete(
            async_call_online_reward_model(url, method, **kwargs)
        )
        return score, extra_infos
    except Exception:
        return 0.0, {}


default_compute_score = async_call_online_reward_model

@register("external")
class ExternalRewardManager:
    """The external reward manager."""

    def __init__(
        self,
        reward_api,
        tokenizer,
        num_examine,
        reward_fn_key="data_source",
        reward_api_method="POST",
        compute_score=None,
    ) -> None:
        self.reward_api = reward_api
        self.reward_api_method = reward_api_method
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = default_compute_score
        self.reward_fn_key = reward_fn_key

    async def async_compute_score(
        self, input_item: ExternalRewardScoreInput
    ) -> ScoreFuncOutput:
        """Asynchronous processing of a single scoring request"""
        try:
            # NOTE: 使用异步的 async_call_online_reward_model 函数，以提高异步性能
            score, extra_info = await self.compute_score(
                url=self.reward_api,
                method=self.reward_api_method,
                data_source=input_item.data_source,
                prompt_str=input_item.prompt_str,
                response_str=input_item.response_str,
                sequence_str=input_item.sequence_str,
                ground_truth=input_item.ground_truth,
                extra_info=input_item.extra_info,
                valid_response_length=input_item.valid_response_length,
            )
            return ScoreFuncOutput(score=score, extra_info=extra_info)
        except Exception as e:
            print(f"Error computing score: {str(e)}")
            return ScoreFuncOutput(score=0.0, extra_info={})

    async def batch_compute_scores(
        self, input_list: List[ExternalRewardScoreInput], max_concurrency: int = 30
    ) -> List[ScoreFuncOutput]:
        """Parallel processing of all scoring requests, limiting the maximum number of concurrent requests"""
        # Create a semaphore to control the number of concurrent requests
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_compute_score(input_item: ExternalRewardScoreInput):
            # NOTE: 使用信号量限制并发
            async with semaphore:
                return await self.async_compute_score(input_item)

        # NOTE: 为每个输入创建任务，并发量会被信号量限制
        tasks = [bounded_compute_score(input_item) for input_item in input_list]

        # NOTE: gather 确保结果顺序与输入顺序一致
        return await asyncio.gather(*tasks)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # NOTE: 如果存在 rm_scores，则直接返回 rm_scores。否则，通过 rm_score_fn 计算
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        score_input_list: List[ExternalRewardScoreInput] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            score_input_list.append(
                ExternalRewardScoreInput(
                    data_source=data_source,
                    prompt_str=prompt_str,
                    response_str=response_str,
                    sequence_str=sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    valid_response_length=valid_response_length,
                )
            )

        # 并行处理所有评分请求
        score_output_list = asyncio.run(
            self.batch_compute_scores(score_input_list)
        )

        # 处理结果
        for i in range(len(data)):
            data_source = score_input_list[i].data_source
            prompt_str = score_input_list[i].prompt_str
            response_str = score_input_list[i].response_str
            sequences_str = score_input_list[i].sequence_str
            valid_response_length = score_input_list[i].valid_response_length
            score = score_output_list[i].score
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # NOTE: 把奖励分数放在最后一个 token 的位置
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[sequences]", sequences_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                    
        for output in score_output_list:
            reward_extra_info.update(output.extra_info)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor
