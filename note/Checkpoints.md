# Checkpoints

## 目录结构

### FSDP

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
├── global_steps_${i}
│   ├── actor
│   │   ├── huggingface       # default save config and tokenizer, save huggingface model if include ``hf_model`` in checkpoint.contents
│   │   └── fsdp_config.json  # FSDP config file, including world_size and fsdp version
│   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
│   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
│   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
│   ├── critic
│   │   ├── huggingface
│   │   └── fsdp_config.json
│   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
│   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
│   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
└── latest_checkpointed_iteration.txt
```

### Megatron

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
├── global_steps_${i}
│   ├── actor
│   │   ├── huggingface  # default save config and tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
│   │   └── dist_ckpt    # save sharded model/optimizer/rng_states, naming the same as Megatron
│   └── critic
│   │   ├── huggingface
│   │   └── dist_ckpt
└── latest_checkpointed_iteration.txt
```
