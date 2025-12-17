set -x

NOW=$(date +%Y%m%d%H%M)
MODEL_NAME=Qwen3-0.6B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export WANDB_DIR=${MODEL_NAME}-GRPO
# export WANDB_PROJECT=${WANDB_DIR}
# export WANDB_EXP=${WANDB_DIR}-${NOW}
# export WANDB_API_KEY=bxxx
# export WANDB_INIT_TIMEOUT=1200

export SWANLAB_DIR=${MODEL_NAME}-GRPO
export SWANLAB_PROJECT=veRL
export SWANLAB_EXP=${SWANLAB_DIR}-${NOW}
export SWANLAB_API_KEY=xxx

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/ray/tmp


DATASETS=/datasets
mcq_train_path=$DATASETS/mcq/train.parquet
mcq_test_path=$DATASETS/mcq/test.parquet
train_files="['$mcq_train_path']"
test_files="['$mcq_test_path']"

model_path=/models/Qwen/$MODEL_NAME
save_model_checkpoint=/experiments/$SWANLAB_EXP

nnodes=1
n_gpus_per_node=8

mini_batch_size_per_gpu=8
mini_batch_size=$(( mini_batch_size_per_gpu * nnodes * n_gpus_per_node ))
examples_per_rollout=$(( mini_batch_size * 4 ))
micro_batch_size_per_gpu=4
# step = 样本数 * (1 - test_ratio) * epoch / examples_per_rollout

max_model_len=32768
max_response_length=4096
max_prompt_length=$(( max_model_len - max_response_length ))

# 这里的 TP 指的是 rollout 时 vllm/sglang 的 tp
TP=2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${examples_per_rollout} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${SWANLAB_EXP} \
    trainer.default_local_dir=${save_model_checkpoint} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    custom_reward_function.path=/workspace/verl/examples/cheyujie/mcq_reward.py \
    custom_reward_function.name=compute_score \
    trainer.total_epochs=5 $@ 2>&1 | tee ${SWANLAB_PROJECT}.log