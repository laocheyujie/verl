docker create \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --shm-size="512g" \
    --cap-add=SYS_ADMIN \
    -v /data/cheyujie/code/verl:/workspace/verl \
    -v /data/cheyujie/models:/models \
    -v /data/cheyujie/datasets:/datasets \
    -v /data/cheyujie/ray:/ray \
    --name verl \
    verlai/verl:vllm011.latest sleep


docker start verl
docker exec -it verl bash