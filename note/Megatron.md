# Megatron Backend

## Actor/Rollout HybridEngine

### init_model

主节点调用 `init_model` 后，从节点接着初始化模型。

- `MegatronPPOActor`: 计算 log prob，更新模型权重
- `vLLMRollout`: 生成


### generate_sequences

- 训练（Actor）时：PP 很重要，因为模型太大，单卡放不下所有层。
- 生成（Rollout）时：PP 效率很低（因为是一个接一个生成的，会有大量气泡等待）。

HybridEngine 在生成阶段，会将模型权重进行“重组”。原本被 PP 切散的层，会被 Gather（聚集）起来。简单来说，它试图在推理时消除 PP，将其转化为 DP，以便所有 GPU 都能并行地去生成数据，而不是像流水线一样排队。

### update_actor

数据在 DP 维上复制，分发到相同 DP 组内的所有 TP/PP 上，但结果仅收集 TP=0 和最后一个 PP 上的结果。


## ReferenceModel

1. 模型初始化: 同 Actor/Rollout 模型初始化，但不初始化 HybridEngine 和 Optimizer。

2. 计算 Ref Log Prob

## CriticWorker and RewardWorker

1. 模型初始化: 同 ReferenceModel 模型初始化，但 CriticWorker 还需要初始化 Optimizer。

2. CriticWorker 计算 Values

3. 更新 Critic

4. 计算 Reward
