# veRL 笔记

## 入口
### 数据预处理
`examples/data_preprocess/demo.py`

### 训练
`verl.trainer.main_ppo.py:run_ppo`


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

## TaskRunner
1. 获取模型本地路径：如果是 HDFS 路径，则下载到本地；否则就直接返回模型路径
2. 获取 Tokenizer：使用 AutoTokenizer，对于没有 pad_token 的 tokenizer，设置 pad_token 为 eos_token，pad_token_id 为 eos_token_id
3. 获取 Processor：使用 AutoProcessor，没有的返回 None
4. 



## 数据流
A：Parquet 文件 --> B：RLHFDataset --> C：DataLoader + collate_fn --> D：DataProto 原始数据 --> E：pop 提取生成数据 --> F：Rollout 生成 --> G：union 合并数据 --> H：奖励计算 --> I：优势计算 --> J：重新计算 log_probs --> K：计算参考 log_probs --> L：计算价值函数 --> M1：更新 critic --> M2：更新 actor --> N：返回训练指标


### Parquet 文件
```py
data_files = "~/data/rlhf/gsm8k/train.parquet"
```

### RLHFDataset
```py
dataset = RLHFDataset(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor
)
```

### DataLoader + collate_fn
```py
dataloader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)
```

### DataProto 原始数据
```py
batch_dict = next(iter(dataloader))  # 返回 dict
batch: DataProto = DataProto.from_single_dict(batch_dict)
```

### pop 提取生成数据
```py
gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
```

### Rollout 生成
```py
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

### union 合并数据
```py
batch = batch.union(gen_batch_output)
```

### 奖励计算
```py
rewards = self.reward_fn(batch)
batch.batch["token_level_rewards"] = rewards
```

### 优势计算
```py
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)
```
核心实现位于：verl/trainer/ppo/core_algos.py

### 重新计算 log_probs
```py
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

### 计算 reference model 的 log_probs
```py
if self.use_reference_policy:
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
```

### 计算 value function
```py
if self.use_critic:
    values = self.critic_wg.compute_values(batch)
    batch = batch.union(values)
```

### 更新 critic
```py
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

### 更新 actor
```py
actor_output = self.actor_rollout_wg.update_actor(batch)
```

### 返回训练指标
```py
actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
metrics.update(actor_output_metrics)
logger.log(data=metrics, step=self.global_steps)
```

## 权重转换
> https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model

### FSDP to HF
.pt -> .safetensor
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```


### Megatron to HF
```bash
python -m verl.model_merger merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```
