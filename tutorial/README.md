## Config

使用方法：./tutorial/test_config.sh


## 自定义奖励函数

1. 接口模板：`verl/tutorial/custom_reward.py`
2. 使用方法：配置文件中指定
```yaml
custom_reward_function:
  path: "path/to/your/reward_module.py"
  name: "custom_reward_fn"  # 可选，默认为 compute_score
```