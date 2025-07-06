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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    # NOTE: 初始化 Ray 集群，配置 CPU 资源和运行时环境变量
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # NOTE: 创建远程 TaskRunner 实例
        # NOTE: TaskRunner 是 Ray 中的一个远程 actor，它将在 Ray 集群上异步执行主要的训练任务
        runner = TaskRunner.remote()
    # NOTE: 异步执行远程任务 runner.run()，并等待其完成
    # NOTE: 通过 ray.get() 阻塞直到远程任务执行完毕，确保整个初始化流程的顺序性
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """
    NOTE: TaskRunner 是 verl 中实现 PPO/GRPO 训练的核心组件，
    它通过将整个 RL 训练流程封装在一个独立的 Ray Actor 中，实现了任务的封装、资源隔离和分布式协调
    """
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        # NOTE: 加载、解析和验证训练任务的配置（使用 OmegaConf），确保所有参数的正确性和一致性
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        # NOTE: 1. 模型下载
        # NOTE: 将模型文件从 HDFS 远程路径复制到本地，确保所有 Worker 都可以访问，使用共享内存（`use_shm`）可以加速模型加载
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # NOTE: 2. Tokenizer 和 Processor 初始化
        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        # NOTE: AutoTokenizer 然后对于没有 pad_token 的 tokenizer，设置 pad_token 为 eos_token，pad_token_id 为 eos_token_id
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        # NOTE: AutoProcessor
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        # NOTE: 3. Worker 类型选择
        # NOTE: 根据配置中指定的 Actor 策略（如 fsdp 或 megatron），动态选择相应的 Worker 类（例如 ActorRolloutRefWorker 和 CriticWorker），并确定使用的 RayWorkerGroup 类型
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # NOTE: 保证 Critic 和 Actor 使用相同的策略
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (ActorRolloutRefWorker,
                                                   AsyncActorRolloutRefWorker,
                                                   CriticWorker)

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # NOTE: 保证 Critic 和 Actor 使用相同的策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import \
                NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker, AsyncActorRolloutRefWorker,
                CriticWorker)

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        # NOTE: 4. 定义角色到 Worker 类的映射
        # NOTE: 创建一个远程的 Ray Actor，将其包装成一个 Ray Actor 类
        # NOTE: 定义 Ray 资源池的规格和角色到资源池的映射，用于 GPU 资源的分配和管理
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        # NOTE: 5. 定义资源池的规格和角色到资源池的映射
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        # NOTE: 6. Reward Model Worker 的初始化
        # NOTE: 加载用于训练和验证的奖励模型
        if config.reward_model.enable:
            # NOTE: 使用奖励模型
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        # NOTE: 7. Reference Policy Worker 的初始化
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            # NOTE: 使用参考模型计算 KL 散度
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        # NOTE: 8. 加载奖励管理器
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        # NOTE: 9. 创建训练和验证数据集 Dataset 类
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        # NOTE: 随机采样 or 顺序采样
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        # NOTE: 10. 创建 RayPPOTrainer 实例，它是管理所有计算资源和训练流程的中央协调器
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        # NOTE: 11. 初始化训练器的 Workers
        # NOTE: 调用 RayPPOTrainer 的 init_workers() 方法，将配置的 Worker 类实例化到 Ray 集群的 GPU 上，为实际计算做准备
        trainer.init_workers()
        # Start the training process.
        # NOTE: 12. 启动训练过程
        # NOTE: 调用 RayPPOTrainer 的 fit() 方法，启动 PPO 训练循环
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    # NOTE: 数据加载与预处理
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        # NOTE: 动态加载自定义的数据集类
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
