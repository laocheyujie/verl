import datetime
import os

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    # NOTE: 获取nccl后端名称
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")


if not torch.distributed.is_initialized():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group(
        backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=600),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )


world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
device_name = get_device_name()

if rank == 0:
    print(f"device_name: {device_name}")

print(f"rank: {rank}, world_size: {world_size}")


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


# config
fsdp_size = int(os.environ.get("FSDP_SIZE", 4))
ulysses_sequence_parallel_size = int(os.environ.get("ULYSSES_SP", 2))

dp = world_size // ulysses_sequence_parallel_size

device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
if rank == 0:
    print("device_mesh: ")
    print(f"{device_mesh}\n{device_mesh.shape}\n{device_mesh.mesh_dim_names}")

print(f"rank {rank}: device_mesh['ddp'].get_local_rank(): {device_mesh['ddp'].get_local_rank()}")
print(f"rank {rank}: device_mesh['fsdp'].get_local_rank(): {device_mesh['fsdp'].get_local_rank()}")
print(f"rank {rank}: device_mesh.get_coordinate(): {device_mesh.get_coordinate()}")
# print(f"rank {rank}: device_mesh['ddp'].get_rank(): {device_mesh['ddp'].get_rank()}")
# print(f"rank {rank}: device_mesh['fsdp'].get_rank(): {device_mesh['fsdp'].get_rank()}")

ulysses_device_mesh = init_device_mesh(
    device_name, mesh_shape=(dp, ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
)
if rank == 0:
    print("ulysses_device_mesh: ")
    print(f"{ulysses_device_mesh}\n{ulysses_device_mesh.shape}\n{ulysses_device_mesh.mesh_dim_names}")
print(f"rank {rank}: ulysses_device_mesh['dp'].get_local_rank(): {ulysses_device_mesh['dp'].get_local_rank()}")
print(f"rank {rank}: ulysses_device_mesh['sp'].get_local_rank(): {ulysses_device_mesh['sp'].get_local_rank()}")
# print(f"rank {rank}: ulysses_device_mesh['dp'].get_rank(): {ulysses_device_mesh['dp'].get_rank()}")
# print(f"rank {rank}: ulysses_device_mesh['sp'].get_rank(): {ulysses_device_mesh['sp'].get_rank()}")


"""
=================== NOTE ===================
- ddp（data parallel）
- fsdp（fully sharded data parallel）

FSDP 组：共享模型参数
DDP 组：共享梯度

example: 8 张 GPU，可以组织成一个形状为 (2, 4) 的 mesh (ddp, fsdp)
- 第一个维度名叫 "ddp"（2 组数据并行）
- 第二个维度名叫 "fsdp"（每组再分 4 个 shard）
"""

"""
=================== RESULTS ===================
`FSDP_SIZE=4 ULYSSES_SP=2 torchrun --standalone --nproc_per_node=8 device_mesh.py`

device_name: cuda

rank: 1, world_size: 8
rank: 4, world_size: 8
rank: 0, world_size: 8
rank: 3, world_size: 8
rank: 2, world_size: 8
rank: 5, world_size: 8
rank: 7, world_size: 8
rank: 6, world_size: 8

device_mesh: 
DeviceMesh('cuda', [[0, 1, 2, 3], [4, 5, 6, 7]], mesh_dim_names=('ddp', 'fsdp'))
(2, 4)
8
('ddp', 'fsdp')

rank 0: device_mesh.get_coordinate(): [0, 0]
rank 1: device_mesh.get_coordinate(): [0, 1]
rank 2: device_mesh.get_coordinate(): [0, 2]
rank 3: device_mesh.get_coordinate(): [0, 3]
rank 4: device_mesh.get_coordinate(): [1, 0]
rank 5: device_mesh.get_coordinate(): [1, 1]
rank 6: device_mesh.get_coordinate(): [1, 2]
rank 7: device_mesh.get_coordinate(): [1, 3]

rank 0: device_mesh['ddp'].get_local_rank(): 0
rank 1: device_mesh['ddp'].get_local_rank(): 0
rank 2: device_mesh['ddp'].get_local_rank(): 0
rank 3: device_mesh['ddp'].get_local_rank(): 0
rank 4: device_mesh['ddp'].get_local_rank(): 1
rank 5: device_mesh['ddp'].get_local_rank(): 1
rank 6: device_mesh['ddp'].get_local_rank(): 1
rank 7: device_mesh['ddp'].get_local_rank(): 1

rank 0: device_mesh['fsdp'].get_local_rank(): 0
rank 1: device_mesh['fsdp'].get_local_rank(): 1
rank 2: device_mesh['fsdp'].get_local_rank(): 2
rank 3: device_mesh['fsdp'].get_local_rank(): 3
rank 4: device_mesh['fsdp'].get_local_rank(): 0
rank 5: device_mesh['fsdp'].get_local_rank(): 1
rank 6: device_mesh['fsdp'].get_local_rank(): 2
rank 7: device_mesh['fsdp'].get_local_rank(): 3

rank 0: device_mesh['ddp'].get_rank(): 0
rank 1: device_mesh['ddp'].get_rank(): 1
rank 2: device_mesh['ddp'].get_rank(): 2
rank 3: device_mesh['ddp'].get_rank(): 3
rank 4: device_mesh['ddp'].get_rank(): 4
rank 5: device_mesh['ddp'].get_rank(): 5
rank 6: device_mesh['ddp'].get_rank(): 6
rank 7: device_mesh['ddp'].get_rank(): 7

rank 0: device_mesh['fsdp'].get_rank(): 0
rank 1: device_mesh['fsdp'].get_rank(): 1
rank 2: device_mesh['fsdp'].get_rank(): 2
rank 3: device_mesh['fsdp'].get_rank(): 3
rank 4: device_mesh['fsdp'].get_rank(): 4
rank 5: device_mesh['fsdp'].get_rank(): 5
rank 6: device_mesh['fsdp'].get_rank(): 6
rank 7: device_mesh['fsdp'].get_rank(): 7


ulysses_device_mesh: 
DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=('dp', 'sp'))
(4, 2)
('dp', 'sp')

rank 0: ulysses_device_mesh['dp'].get_local_rank(): 0
rank 1: ulysses_device_mesh['dp'].get_local_rank(): 0
rank 2: ulysses_device_mesh['dp'].get_local_rank(): 1
rank 3: ulysses_device_mesh['dp'].get_local_rank(): 1
rank 4: ulysses_device_mesh['dp'].get_local_rank(): 2
rank 5: ulysses_device_mesh['dp'].get_local_rank(): 2
rank 6: ulysses_device_mesh['dp'].get_local_rank(): 3
rank 7: ulysses_device_mesh['dp'].get_local_rank(): 3

rank 0: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 1: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 2: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 3: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 4: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 5: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 6: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 7: ulysses_device_mesh['sp'].get_local_rank(): 1

rank 0: ulysses_device_mesh['dp'].get_rank(): 0
rank 1: ulysses_device_mesh['dp'].get_rank(): 1
rank 2: ulysses_device_mesh['dp'].get_rank(): 2
rank 3: ulysses_device_mesh['dp'].get_rank(): 3
rank 4: ulysses_device_mesh['dp'].get_rank(): 4
rank 5: ulysses_device_mesh['dp'].get_rank(): 5
rank 6: ulysses_device_mesh['dp'].get_rank(): 6
rank 7: ulysses_device_mesh['dp'].get_rank(): 7

rank 0: ulysses_device_mesh['sp'].get_rank(): 0
rank 1: ulysses_device_mesh['sp'].get_rank(): 1
rank 2: ulysses_device_mesh['sp'].get_rank(): 2
rank 3: ulysses_device_mesh['sp'].get_rank(): 3
rank 4: ulysses_device_mesh['sp'].get_rank(): 4
rank 5: ulysses_device_mesh['sp'].get_rank(): 5
rank 6: ulysses_device_mesh['sp'].get_rank(): 6
rank 7: ulysses_device_mesh['sp'].get_rank(): 7

"""


"""
`GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 FSDP_SIZE=4 ULYSSES_SP=2 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=172.16.16.4 --master_port=29501 device_mesh.py`
`GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 FSDP_SIZE=4 ULYSSES_SP=2 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=172.16.16.4 --master_port=29501 device_mesh.py`

device_name: cuda
rank: 0, world_size: 16
rank: 1, world_size: 16
rank: 2, world_size: 16
rank: 3, world_size: 16
rank: 4, world_size: 16
rank: 5, world_size: 16
rank: 6, world_size: 16
rank: 7, world_size: 16

device_mesh: 
DeviceMesh('cuda', [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], mesh_dim_names=('ddp', 'fsdp'))
(4, 4)
16
('ddp', 'fsdp')

rank 0: device_mesh['ddp'].get_local_rank(): 0
rank 1: device_mesh['ddp'].get_local_rank(): 0
rank 2: device_mesh['ddp'].get_local_rank(): 0
rank 3: device_mesh['ddp'].get_local_rank(): 0
rank 4: device_mesh['ddp'].get_local_rank(): 1
rank 5: device_mesh['ddp'].get_local_rank(): 1
rank 6: device_mesh['ddp'].get_local_rank(): 1
rank 7: device_mesh['ddp'].get_local_rank(): 1

rank 0: device_mesh['fsdp'].get_local_rank(): 0
rank 1: device_mesh['fsdp'].get_local_rank(): 1
rank 2: device_mesh['fsdp'].get_local_rank(): 2
rank 3: device_mesh['fsdp'].get_local_rank(): 3
rank 4: device_mesh['fsdp'].get_local_rank(): 0
rank 5: device_mesh['fsdp'].get_local_rank(): 1
rank 6: device_mesh['fsdp'].get_local_rank(): 2
rank 7: device_mesh['fsdp'].get_local_rank(): 3


ulysses_device_mesh: 
DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]], mesh_dim_names=('dp', 'sp'))
(8, 2)
('dp', 'sp')

rank 0: ulysses_device_mesh['dp'].get_local_rank(): 0
rank 1: ulysses_device_mesh['dp'].get_local_rank(): 0
rank 2: ulysses_device_mesh['dp'].get_local_rank(): 1
rank 3: ulysses_device_mesh['dp'].get_local_rank(): 1
rank 4: ulysses_device_mesh['dp'].get_local_rank(): 2
rank 5: ulysses_device_mesh['dp'].get_local_rank(): 2
rank 6: ulysses_device_mesh['dp'].get_local_rank(): 3
rank 7: ulysses_device_mesh['dp'].get_local_rank(): 3

rank 0: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 1: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 2: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 3: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 4: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 5: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 6: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 7: ulysses_device_mesh['sp'].get_local_rank(): 1

node-2:
rank: 9, world_size: 16
rank: 8, world_size: 16
rank: 10, world_size: 16
rank: 11, world_size: 16
rank: 12, world_size: 16
rank: 13, world_size: 16
rank: 14, world_size: 16
rank: 15, world_size: 16

rank 8: device_mesh['ddp'].get_local_rank(): 2
rank 9: device_mesh['ddp'].get_local_rank(): 2
rank 10: device_mesh['ddp'].get_local_rank(): 2
rank 11: device_mesh['ddp'].get_local_rank(): 2
rank 12: device_mesh['ddp'].get_local_rank(): 3
rank 13: device_mesh['ddp'].get_local_rank(): 3
rank 14: device_mesh['ddp'].get_local_rank(): 3
rank 15: device_mesh['ddp'].get_local_rank(): 3

rank 8: device_mesh['fsdp'].get_local_rank(): 0
rank 9: device_mesh['fsdp'].get_local_rank(): 1
rank 10: device_mesh['fsdp'].get_local_rank(): 2
rank 11: device_mesh['fsdp'].get_local_rank(): 3
rank 12: device_mesh['fsdp'].get_local_rank(): 0
rank 13: device_mesh['fsdp'].get_local_rank(): 1
rank 14: device_mesh['fsdp'].get_local_rank(): 2
rank 15: device_mesh['fsdp'].get_local_rank(): 3


rank 8: ulysses_device_mesh['dp'].get_local_rank(): 4
rank 9: ulysses_device_mesh['dp'].get_local_rank(): 4
rank 10: ulysses_device_mesh['dp'].get_local_rank(): 5
rank 11: ulysses_device_mesh['dp'].get_local_rank(): 5
rank 12: ulysses_device_mesh['dp'].get_local_rank(): 6
rank 13: ulysses_device_mesh['dp'].get_local_rank(): 6
rank 14: ulysses_device_mesh['dp'].get_local_rank(): 7
rank 15: ulysses_device_mesh['dp'].get_local_rank(): 7

rank 8: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 9: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 10: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 11: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 12: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 13: ulysses_device_mesh['sp'].get_local_rank(): 1
rank 14: ulysses_device_mesh['sp'].get_local_rank(): 0
rank 15: ulysses_device_mesh['sp'].get_local_rank(): 1

"""
