# veRL 笔记

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

