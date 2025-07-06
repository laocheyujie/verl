export CUDA_VISIBLE_DEVICES="4,5,6,7"
export WANDB_API_KEY=
export HYDRA_FULL_ERROR=1

set -x
num_gpus=4
data_dir=/data/cheyujie/github_fork/verl/data/aminer
model_path=/data/cheyujie/code/all_in_one/models/huggingface/Qwen/Qwen3-8B
project_name=VERL-External
cur_task=test-external-qwen3-8b
save_model_checkpoint=/data/cheyujie/github_fork/verl/train_models/$cur_task

ray stoexport CUDA_VISIBLE_DEVICES="0,2,3,4"
export WANDB_API_KEY=
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/data/cheyujie/ray/tmp
export PYTHONPATH=/data/cheyujie/github_fork/verl:$PYTHONPATH
export VLLM_INITIALIZATION_MAX_GPU_UTILIZATION=0.4
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION

set -x

nproc_per_gpu=1
nnodes=1
ngpu_per_node=4

total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
mini_batch_size=$(( total_procs ))

data_dir=/data/cheyujie/github_fork/verl/data
model_path=/data/cheyujie/code/all_in_one/models/huggingface/Qwen/Qwen3-0.6B
project_name=VERL-External
cur_task=test-external-qwen3-8b
save_model_checkpoint=/data/liuxinyu/projects/verl/dagang/train_models/$cur_task

ray stop
ray start --head --node-ip-address=0.0.0.0 --port=6378 --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external --num-cpus 16 --num-gpus $num_gpus


echo "启动训练..."

# actor_rollout_ref.actor.use_dynamic_bsz=True \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=${total_procs} \
    data.max_prompt_length=12800 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16896 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16896 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    reward_model.reward_manager=external \
    +reward_model.reward_api=http://127.0.0.1:8018/reward \
    +reward_model.reward_api_method=POST \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$cur_task \
    trainer.n_gpus_per_node=${ngpu_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=20 \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
p
ray start --head --node-ip-address=0.0.0.0 --port=6378 --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external --num-cpus 16 --temp-dir=/data3/ray --num-gpus $num_gpus


echo "启动训练..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=20 \
    data.max_prompt_length=32768 \
    data.max_response_length=4096 \
    data.truncation=right \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=20 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=external \
    +reward_model.reward_api=http://127.0.0.1:8018/reward \
    +reward_model.reward_api_method=POST \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$cur_task \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
