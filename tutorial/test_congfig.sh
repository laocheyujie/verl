# -*- coding: utf-8 -*-
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NOW=$(date +%Y%m%d%H%M)
MODEL_NAME=Qwen3-8B

export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/data/cheyujie/ray/tmp


data_dir=/data/cheyujie/github_fork/verl/data/aminer
model_path=/data/cheyujie/code/all_in_one/models/huggingface/Qwen/${MODEL_NAME}
save_model_checkpoint=/data/cheyujie/experiments/$WANDB_EXP


set -x

nproc_per_gpu=32
nnodes=1
ngpu_per_node=8
num_gpus=$(( nnodes * ngpu_per_node ))
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
mini_batch_size=$(( total_procs ))
micro_batch_size=$(( mini_batch_size / ngpu_per_node ))

max_prompt_length=32768
max_response_length=4096
max_model_len=$(( max_prompt_length + max_response_length ))

TP=4
SP=2

echo "启动训练..."

# actor_rollout_ref.actor.use_dynamic_bsz=True \
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \

python3 -m verl.trainer.get_config \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=${total_procs} \
    data.val_batch_size=${total_procs} \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=$max_prompt_length \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_model_len \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=external \
    +reward_model.reward_api=http://127.0.0.1:8018/reward \
    +reward_model.reward_api_method=POST \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.n_gpus_per_node=${ngpu_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
