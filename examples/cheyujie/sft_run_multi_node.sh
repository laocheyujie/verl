set -x

nnodes=2  # 机器数量
nproc_per_node=8  # 每台机器上的卡数量
CONFIG_PATH="examples/cheyujie/sft.yaml"  # 这里一定要修改成你自己机器上的绝对路径
MAIN_NODE_IP=00.00.00.00  # 主节点的 ip
node_rank=0  # 当前机器的 rank
port=8324

python3 -m torch.distributed.run --nnodes=$nnodes --nproc_per_node=$nproc_per_node \
    --node_rank=$node_rank --master_addr=$MAIN_NODE_IP --master_port=$port \
    -m verl.trainer.fsdp_sft_trainer \
    --config_path=$CONFIG_PATH

# ./examples/cheyujie/sft_run_multi_node.sh 2>&1 | tee -a train.log