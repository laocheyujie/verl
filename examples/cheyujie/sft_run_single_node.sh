set -x

CONFIG_PATH="examples/cheyujie/sft.yaml"  # 刚刚写的脚本，这里应该写绝对路径

nproc_per_node=8

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=65536 \
    -m verl.trainer.fsdp_sft_trainer \
    --config_path=$CONFIG_PATH

# ./examples/cheyujie/sft_run_single_node.sh 2>&1 | tee -a train.log