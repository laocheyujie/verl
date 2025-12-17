# veRL 笔记

## 入口

### 数据预处理

`examples/data_preprocess/demo.py`

### 训练

`verl.trainer.main_ppo.py:run_ppo`

## RLHFDataset

`verl/utils/dataset/rl_dataset.py`

1. 支持从远程存储下载 Parquet 文件到本地缓存，支持共享内存加速文件访问，自动管理文件路径，支持检查点恢复。
2. 使用 HuggingFace `datasets` 库读取 Parquet 文件，支持多个数据文件的合并，自动处理数据格式转换。
3. 根据最大长度过滤过长的 prompts，支持多进程并行处理，可配置的过滤策略。
4. 支持图像和视频的多模态输入，解析 `<image>` 和 `<video>` 标签，将多模态内容转换为结构化格式。
5. 添加 chat template 来格式化对话，将文本转换为 token IDs，生成 attention mask 和 position ids。
6. padding 到指定长度，支持多种截断策略（left, right, middle, error），生成位置编码。
7. 支持训练中断后的恢复，可以从原始文件重新构建数据集，兼容序列化/反序列化。
8. 返回包含以下关键字段的字典：`input_ids`, `attention_mask`, `position_ids`, `raw_prompt_ids`, `multi_modal_data`, `multi_modal_inputs`, `index`, `tools_kwargs`。

`tools_kwargs` 结构如下：

```py
tools_kwargs = {
    "tool_name": {
        "create_kwargs": {...},      # 工具创建时的参数
        "execute_kwargs": {...},     # 工具执行时的参数（可选）
        "calc_reward_kwargs": {...}, # 计算奖励时的参数（可选）
        "release_kwargs": {...},     # 释放资源时的参数（可选）
    }
}
```

## run_ppo

```py
def run_ppo(config) -> None:
    # 初始化 Ray 集群，配置 CPU 资源和运行时环境变量
    ray.init(
        runtime_env={"env_vars": {...}},
        num_cpus=config.ray_init.num_cpus,
    )

    # 创建远程 TaskRunner 实例
    # TaskRunner 是 Ray 中的一个远程 actor，它将在 Ray 集群上异步执行主要的训练任务
    runner = TaskRunner.remote()
    # 异步执行远程任务 runner.run()，并等待其完成
    # 通过 ray.get() 阻塞直到远程任务执行完毕，确保整个初始化流程的顺序性
    ray.get(runner.run.remote(config))
```

## TaskRunner.run

1. 加载配置
2. 定义 Worker 类、Actor 类、角色到 Worker 映射、角色到资源池映射
   1. 添加 Actor Rollout (`add_actor_rollout_worker`)
      1. 选择 Worker 类：`ActorRolloutRefWorker` / `AsyncActorRolloutRefWorker`
      2. 创建远程 Ray Actor：`ray.remote(actor_rollout_cls)`
      3. 定义 Ray 角色到 Worker 映射：`self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)`
   2. 添加 Critic (`add_critic_worker`)
      1. 选择 Worker 类：`CriticWorker`
      2. 创建远程 Ray Actor：`ray.remote(CriticWorker)`
      3. 定义 Ray 角色到 Worker 映射：`self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)`
   3. 添加奖励模型 (`add_reward_model_worker`)
      1. 选择 Worker 类：`RewardModelWorker`
      2. 创建远程 Ray Actor：`ray.remote(RewardModelWorker)`
      3. 定义 Ray 角色 (RewardModel) 到 Worker 映射：`self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)`
      4. 定义 Ray 角色 (RewardModel) 到资源池映射：`self.mapping[Role.RewardModel] = "reward_pool"` 或 `self.mapping[Role.RewardModel] = "global_pool"`
   4. 添加参考策略 (`add_ref_policy_worker`)
      1. 复用 Actor Rollout Worker 类
      2. 创建远程 Ray Actor：`ray.remote(ref_policy_cls)`
      3. 定义 Ray 角色 (RefPolicy) 到 Worker 映射：`self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)`
      4. 定义 Ray 角色 (RefPolicy) 到资源池映射：`self.mapping[Role.RefPolicy] = "global_pool"`
   5. 验证配置
3. 获取模型本地路径：如果是 HDFS 路径，则下载到本地；否则就直接返回模型路径
4. 获取 Tokenizer：使用 AutoTokenizer，对于没有 pad_token 的 tokenizer，设置 pad_token 为 eos_token，pad_token_id 为 eos_token_id
5. 获取 Processor：使用 AutoProcessor，没有的返回 None
6. 加载奖励管理器
7. 初始化 Ray 资源池管理器
   1. 定义资源池到资源的映射：`resource_pool_spec = {"global_pool": [8, 8], "reward_pool": [...]}`
   2. 定义 Ray 角色 (ActorRollout) 到资源池映射：`self.mapping[Role.ActorRollout] = "global_pool"`
   3. 定义 Ray 角色 (Critic) 到资源池映射：`self.mapping[Role.Critic] = "global_pool"`
   4. 实例化资源管理器 `resource_pool_manager = ResourcePoolManager(resource_pool_spec, self.mapping)`
8. 创建训练和验证数据集 Dataset 类
9. 创建 RayPPOTrainer 实例 `trainer`：它是管理所有计算资源和训练流程的中央协调器
10. 初始化训练器的 Workers：`trainer.init_workers()`
    1. 创建资源池：通过 `ResourcePoolManager` 创建 Ray 资源池，得到 `self.resource_pool_dict`
    2. 初始化资源池到类的映射：为每个资源池创建一个字典，用于存储不同角色 Worker 的 `RayClassWithInitArgs` 包装器，`RayClassWithInitArgs` 用于延迟初始化 Worker，存储了 Worker 的类和初始化参数，先初始化空的 `self.resource_pool_to_cls`
    3. 创建不同角色的 Worker 的 `RayClassWithInitArgs` 实例，存到 `self.resource_pool_to_cls` 里
    4. 初始化 WorkerGroup：遍历所有资源池，将同一资源池中的多个 Worker 类通过 `create_colocated_worker_cls` 组合成一个共置类，然后实例化 `RayWorkerGroup`，获取每个角色对应的 Actor 实例
    5. 根据角色从 `all_wg` 字典中获取对应的 `WorkerGroup`，并调用其 `init_model()` 方法依次初始化 critic, ref, rm, actor_rollout 对应的 `Worker` 模块
11. 启动训练 ` trainer.fit()`

## ActorRolloutRefWorker

### \_\_init\_\_

1. 初始化 PyTorch 分布式环境
2. 为 FSDP 创建设备网格，用于模型参数的分片 `self.device_mesh` `(ddp, fsdp)`
   - ddp_size = world_size // fsdp_size
   - 第一维用于复制（Replication，像 DDP），第二维用于分片（Sharding，像 FSDP）
   - HYBRID_SHARD: 节点间 (Inter-node) -> DDP; 节点内 (Intra-node) -> FSDP
3. 为 SP 创建设备网格 `self.ulysses_device_mesh` `(dp, sp)`
   - dp_size = world_size // sp_size
4. 初始化 Ulysses 分片管理器
5. 标记是否使用 LoRA
6. 根据 Worker 角色配置 profiler，用于性能分析配置
7. 配置 actor 和 ref 的 parameter offload 和 optimizer offload
8. 规范化 actor 相关配置 (`actor.ppo_mini_batch_size` 和 `actor.ppo_micro_batch_size_per_gpu`)
   - `ppo_mini_batch_size`: 模型会在数据累积到一个 mini_batch 后更新一次参数
     1. `ppo_mini_batch_size *= rollout.n` 得到更新一次参数使用的**全局**总样本数
     2. `ppo_mini_batch_size = ppo_mini_batch_size // (world_size // ulysses_sequence_parallel_size)` (实际上 `world_size // ulysses_sequence_parallel_size = dp`) 得到**每个 dp 组**分到的 mini_batch 数
   - `ppo_micro_batch_size`: 梯度累积使用的最小样本数
     1. `ppo_micro_batch_size_per_gpu = ppo_micro_batch_size // (world_size // ulysses_sequence_parallel_size)` (实际上 `world_size // ulysses_sequence_parallel_size = dp`) 得到**每个 dp 组**分到的 micro_batch 数
   - 如果想直接设置梯度累计的最小样本数，直接设置 `ppo_micro_batch_size_per_gpu`
9. 规范化 rollout 相关配置 (`rollout.log_prob_micro_batch_size_per_gpu`)
10. 规范化 ref 相关配置 (`ref.log_prob_micro_batch_size_per_gpu`)

**配置里的相关概念**：

- `ppo_mini_batch_size`: 更新一次参数消耗的全局样本数 (不含 \*n)
- `ppo_micro_batch_size`: 实际单次前向传播消耗的全局样本数 (做梯度累积)
- `ppo_micro_batch_size_per_gpu`: 实际单次前向传播每个 dp 组消耗的样本数 (做梯度累积和 dp)

配置中：

- mini_batch_size = mini_batch_size_per_gpu \* nnodes \* n_gpus_per_node
- 每一次 rollout 消耗的样本数 = mini_batch_size \* 倍率
- step = num_data \* (1 - test_ratio) \* epoch / 每一次 rollout 消耗的样本数

### \_build_model_optimizer

1. 加载 `self.tokenizer` 和 `self.processor`
2. 如有 `custom_chat_template` 则替换 `.chat_template` 为 `custom_chat_template`
3. `torch_dtype`: Actor 使用 fp32; Ref 使用 bf16
4. 加载模型配置 `actor_model_config = AutoConfig.from_pretrained(...)`
5. 加载生成配置 `self.generation_config = GenerationConfig.from_pretrained(...)`
6. 获取 `init_context`: 在创建一个巨大的模型时，先不为模型的参数分配任何实际的内存，而是创建一个“空壳”模型
7. 根据模型配置选择对应的 `AutoModelForxxx`
8. 加载模型 `AutoModelForxxx.from_pretrained(...)`
   1. 由于在 `init_context` 下面，因此只是加载模型空壳
   2. Actor model 先以 fp32 加载，这样 optimizer 根据参数会自动初始化为 fp32，最后再把模型参数转为 bf16
9. 应用 liger kernel 和 fused kernel 优化技术
10. 应用 gradient checkpointing 优化技术
11. 应用 LoRA
12. 冻结视觉模型（如需），把视觉模型部分的梯度设为 Fasle
13. `torch.distributed.barrier()`
14. 设定混合精度配置下每部分的具体精度
15. 用 FSDP 包裹模型
    1. 参数类型设为 bf16
    2. role 只决定 offload 相关配置，都会用 FSDP
    3. 获取 `wrap_policy`，用于指导 FSDP 在遍历模型的每个模块时是否符合规则从而被包装，被包装的模块的参数、梯度、优化器都会被分散存储
    4. 如果是 fsdp 策略，直接用 `FSDP` 包装 `actor_module`
    5. 如果是 fsdp2 策略
       1. `full_state = actor_module.state_dict()` 在 CPU 内存中创建了整个模型的一个完整副本，不会导致 GPU 显存增加，但会导致 CPU 内存的增加
          - 在初始化时，Rank 0 使用了 `cpu_init_weights` 上下文，在 CPU 内存中加载了完整的、真实的预训练模型权重
          - `full_state` 是一个包含所有参数名和对应 Tensor 数值的字典 (dict)
          - `full_state` 仅被 Rank 0 用作数据源
       2. `apply_fsdp2` 定义分布式切分规则
          - 将普通的 `nn.Module` 转换为支持参数分片（`Sharding`）的分布式模块
          - 它定义了参数该如何被切分、使用什么混合精度（Mixed Precision）以及是否开启 CPU Offload
          - 此时的 `model` 在逻辑上已经不再是一个拥有完整参数的普通模型，而是一个 FSDP Module
          - 对于 FSDP Module，当调用 `.to(device)` 时，PyTorch 只会将属于当前 Rank 的本地分片（Local Shard）移动到目标设备（GPU）
          - 模型内部的参数可能还是 Meta Tensor（空壳）或者未正确同步
       3. `fsdp2_load_full_state_dict` 加载并广播权重，从主节点分发权重数据到各个分片
          - Rank 0 不仅要加载属于自己的那一部分参数，还负责将其他 Rank 需要的参数分片（Shard）切分出来，并通过网络广播（Broadcast）发送给对应的 Rank
       4. 如果启用了激活卸载，则启用它
       5. 如果当前 Worker 是 Actor 角色，则初始化 AdamW 优化器和学习率调度器

### \_build_rollout

1. 解析配置
2. 构建设备网格 (Device Mesh)
   - 将所有的 GPU 划分为两维网格：
     1. `dp` (Data Parallel): 数据并行维度，不同的 DP 组处理不同的数据（Prompt）
     2. `infer_tp` (Tensor Parallel): 推理时的张量并行维度，一个模型被切分到 `infer_tp` 个 GPU 上同时运行
3. 如果是 vLLM/SGLang 等，设置 `is_collect`: 在 Tensor Parallel 中，只有 rank 0 负责收集最终生成的 token 结果，其他 rank 只是协助计算。这个标志位用于后续告诉系统哪些进程需要回传生成的数据
4. 初始化随机状态：确保生成的随机性独立且可控，同时不影响训练的随机状态
5. 构建 Rollout 模型：根据配置实例化具体的 Rollout 引擎 `self.rollout`
6. 设置 FSDP 状态字典类型：当从 FSDP 模型中加载或保存权重用于推理时，需要指定权重的聚合方式
   - 单卡：使用完整权重 (FULL_STATE_DICT)
   - 多卡：通常使用分片权重 (SHARDED_STATE_DICT) 以节省内存，避免在单张卡上聚合整个模型导致 OOM
7. 切换到 Trainer 模式，确保在开始任何操作前，权重加载的准备工作已经就绪
   - FSDP 模型 (Trainer 侧)：是权重的真理之源，所有的 Checkpoint 加载、保存、优化器更新，都是针对 FSDP 模型进行的
   - Rollout 引擎 (Inference 侧)：通常是下游消费者，它需要从 FSDP 模型那里“同步”权重，或者共享内存
   - 当 `_build_rollout` 被调用时，后续流程顺序如下：
     1. 初始化 Worker
        - 构建 FSDP 模型（Trainer 侧）
        - 构建 Rollout 引擎（Inference 侧）
        - 强制切到 `trainer_mode`
     2. 加载预训练权重 (Load Checkpoint)
        - 必须在 Trainer Mode 下进行
        - 外部控制器调用 `load_state_dict`
        - 权重数据从磁盘读取 -> 填入 FSDP 模型 (Trainer) 中
     3. 同步权重：从 Trainer -> Rollout
        - 通常发生在每一轮 PPO 迭代开始生成之前
        - 系统将 FSDP 模型中刚刚加载好的（或者刚训练更新过的）权重，复制/广播给 Rollout 引擎
     4. 再切换模式为 `rollout_mode`
        - 如果是混合部署（单卡同时做训练和推理），可能涉及显存置换（把 FSDP 模型移到 CPU/内存，给 Rollout 腾地）
     5. 生成 `generate()`
        - Rollout 引擎使用刚刚同步过来的最新权重进行推理

## RayPPOTrainer

### fit

1. 创建 Tracking 日志记录器
2. 设置全局步数
3. 加载模型检查点和数据集 dataloader
   1. 获取检查点步数
      - `config.trainer.resume_mode == "auto"`: `global_steps` 为 `latest_checkpointed_iteration.txt` 记录的值
      - `config.trainer.resume_mode == "resume_path"`: `global_steps` 为 `resume_from_path` 的 `global_step_xxx` 值
   2. 加载 actor 权重
   3. 加载 critic 权重 (如果设置了 `use_critic`)
   4. 加载 dataloader
4. `for epoch in range(total_epochs)` 遍历配置的总 epoch 数
5. `for batch in self.train_dataloader` 每个 epoch 内再遍历 train_dataloader
6. 从 batch 中分离出用于 rollout 的数据 (`input_ids`, `attention_mask`, `position_ids`)，保留其他数据用于后续处理
7. 为每个样本分配唯一 ID，重复数据 `config.actor_rollout_ref.rollout.n` 次以对齐多次采样
8. 调用 `ActorRolloutWorker` 生成序列，并记录生成时间
9. 处理 REMAX 基线 (如果 `adv_estimator == 'remax'`)：生成确定性基线序列，计算基线奖励，用于 REMAX 优势估计器
10. 计算响应掩码 `response_mask`，并可选地进行批次平衡
11. 根据配置使用奖励模型或自定义奖励函数计算 token 级别的奖励分数 `reward_tensor`，支持同步和异步计算
12. 使用 megatron 基于训练开始前的 policy 重新计算 behaviour policy 的 `old_log_probs`，用于**重要性采样**，同时计算熵值
13. 使用 Reference policy 计算 `ref_log_probs`，用于 **KL 散度计算**
14. 使用 Critic 网络计算状态价值 `values`，用于**优势函数估计**
15. 根据配置的优势估计器 (GAE、GRPO、REMAX 等) 计算优势函数 `adv`，支持 KL 惩罚
16. 使用计算出的优势函数更新 `Critic` 网络参数
17. 在 Critic 预热完成后，使用 PPO 损失函数更新 `Actor` 网络参数 (`actor_rollout_wg.update_actor(batch)` -> `actor.update_policy(data=data)`)
    1. 第一层 ppo_epochs 循环，代表 off-policy，即一个 batch 的经验被用来更新多少次模型
    2. 第二层循环将 train_batch_size 按 ppo_mini_batch_size 切分，每个 ppo_mini_batch 更新一次参数
    3. 第三层循环将 ppo_mini_batch 拆成更小的 micro_batch，做多次前向和反向累积好梯度后更新一次参数
18. 将生成的序列、输入、输出和分数保存到指定目录
19. 根据配置的频率执行验证，计算验证指标并记录
20. 根据配置的频率保存模型检查点
    1. 保存 actor 权重
    2. 保存 critic 权重 (如果设置了 `use_critic`)
    3. 保存 dataloader 的 state_dict
    4. 把当前的步数写入 `latest_checkpointed_iteration.txt`
21. 收集训练指标、时序指标和吞吐量指标，并记录到日志系统

### 训练数据流

1. Parquet 文件
2. RLHFDataset
3. DataLoader + collate_fn
4. DataProto 原始数据
5. pop 提取生成数据
6. Rollout 生成
7. union 合并数据
8. 奖励计算
9. 优势计算
10. 重新计算 old_log_probs
11. 计算参考 ref_log_probs
12. 计算价值函数 values
13. 更新 critic
14. 更新 actor
15. 返回训练指标

```py
# Parquet 文件
data_files = "~/data/rlhf/gsm8k/train.parquet"

# RLHFDataset
dataset = RLHFDataset(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor
)

# DataLoader + collate_fn
dataloader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

# DataProto 原始数据
batch_dict = next(iter(dataloader))  # 返回 dict
batch: DataProto = DataProto.from_single_dict(batch_dict)

# pop 提取生成数据
gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

# Rollout 生成
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

# union 合并数据
batch = batch.union(gen_batch_output)

# 计算奖励分数 rewards
rewards = self.reward_fn(batch)
batch.batch["token_level_rewards"] = rewards

# 重新计算 old_log_probs
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)

# 计算 reference model 的 ref_log_probs
if self.use_reference_policy:
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)

# 计算价值函数 values
if self.use_critic:
    values = self.critic_wg.compute_values(batch)
    batch = batch.union(values)

# 优势计算
# 核心实现位于：verl/trainer/ppo/core_algos.py
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)

# 更新 critic
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)

# 更新 actor
actor_output = self.actor_rollout_wg.update_actor(batch)

# 返回训练指标
actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
metrics.update(actor_output_metrics)
logger.log(data=metrics, step=self.global_steps)
```

## 变量

### self.role_worker_mapping

```py
{
    Role.ActorRollout: ray.remote(ActorRolloutRefWorker), # 或 ray.remote(AsyncActorRolloutRefWorker)
    Role.Critic: ray.remote(CriticWorker),
    Role.RewardModel: ray.remote(RewardModelWorker),
    Role.RefPolicy: ray.remote(ActorRolloutRefWorker) # 或 ray.remote(AsyncActorRolloutRefWorker)
}
```

### self.mapping

```py
{
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
    Role.RewardModel: "reward_pool", # 或 "global_pool"
    Role.RefPolicy: "global_pool"
}
```

### resource_pool_spec

资源池名称和对应的卡

```py
# 假定共有 32卡
{
    "global_pool": [8, 8],
    "reward_pool": [8, 8]
}
```

### self.resource_pool_dict

资源池名称和对应的资源池实例

```py
{
    "global_pool": RayResourcePool(...),
    "reward_pool": RayResourcePool(...)
}
```

### self.resource_pool_to_cls

资源池实例和对应的 Worker 角色和他们的初始化参数信息

```py
{
    RayResourcePool(...): {
        "actor_rollout": RayClassWithInitArgs(...),
        "critic": RayClassWithInitArgs(...),
        "ref": RayClassWithInitArgs(...)
    },
    RayResourcePool(...): {
        "rm": RayClassWithInitArgs(...)
    }
}
```

## 参考资料

- [HybridFlow / veRL 原文浅析](https://zhuanlan.zhihu.com/p/24682036412)
- [深入浅出理解 verl 源码（Part 1）](https://zhuanlan.zhihu.com/p/1920751852749849692)
- [深入浅出理解 verl 源码——Rollout](https://zhuanlan.zhihu.com/p/1923349757566388159)
