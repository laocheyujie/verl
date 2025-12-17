set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

NOW=$(date +%Y%m%d%H%M)
MODEL_NAME=Qwen3-0.6B

export SWANLAB_DIR=${MODEL_NAME}-GRPO
export SWANLAB_PROJECT=veRL
export SWANLAB_EXP=${SWANLAB_DIR}-${NOW}

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/ray/tmp

DATASETS=/datasets
mcq_train_path=$DATASETS/mcq/train.parquet
mcq_test_path=$DATASETS/mcq/test.parquet
train_files="['$mcq_train_path']"
test_files="['$mcq_test_path']"

hf_model_path=/models/Qwen/$MODEL_NAME
mcore_model_path=/models/Qwen/${MODEL_NAME}-mcore
save_model_checkpoint=/experiments/$SWANLAB_EXP

nnodes=2
n_gpus_per_node=8
num_gpus=$(( nnodes * n_gpus_per_node ))

mini_batch_size_per_gpu=8
mini_batch_size=$(( mini_batch_size_per_gpu * nnodes * n_gpus_per_node ))
examples_per_rollout=$(( mini_batch_size * 2 ))
micro_batch_size_per_gpu=4

max_model_len=32768
max_response_length=4096
max_prompt_length=$(( max_model_len - max_response_length ))

PP=2
TP=2
# EP=2
# ETP=1
VLLM_TP=2

# Offload
ALL_OFFLOAD=${ALL_OFFLOAD:-False}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=examples/cheyujie/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$examples_per_rollout \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$hf_model_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$mcore_model_path \
    actor_rollout_ref.actor.megatron.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.megatron.grad_offload=$ACTOR_GRAD_OFFLOAD \
    actor_rollout_ref.actor.megatron.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.param_offload=$REF_PARAM_OFFLOAD \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$mcore_model_path \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${SWANLAB_EXP} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    custom_reward_function.path=examples/cheyujie/mcq_reward.py \
    custom_reward_function.name=compute_score \
    trainer.total_epochs=1 $@ 2>&1 | tee ${SWANLAB_PROJECT}.log
