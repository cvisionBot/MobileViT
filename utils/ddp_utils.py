# Public Lib
import socket
import torch
import torch.distributed as dist

from utils import logger

'''
- torch.distributed module
    PyTorch의 분산 패키지로 연구자와 실무자가 여러 프로세스와 클러스터의 기기에서 계산을 쉽게 병렬화 할 수 있게 한다.
    이를 위해 각 프로세스가 다른 프로세스와 데이터를 교환할 수 있도록 메시지 교환 규약 (messaging passing semantics) 활용한다.
'''

def is_master(opts):
    node_rank = getattr(opts, "ddp.rank", 0)
    return node_rank == 0

'''
- python getaatr(object, name[, default]) object에 존해나는 속성 값을 가져온다.
'''

def dist_barrier():
    dist.barrier()

'''
- torch.distributed.barrier(group=None, async_op=False, device_ids=None)
    모든 프로세스의 동기화를 담당한다.
'''

def is_start_rank_node(opts):
    node_rank = getattr(opts, "ddp.rank", 0)
    def_rank = getattr(opts, "ddp.start_rank", 0)
    return node_rank == def_rank


def distributed_init(opts):
    ddp_url = getattr(opts, "ddp.dist_url", None)
    is_master_node = is_master(opts)
    if ddp_url is None:
        ddp_port = getattr(opts, "ddp.dist_port", 6006)
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)
    if torch.distributed.is_initialized():
        logger.warning("DDP is already initialized and cannot be initialize twice!")
    else:
        logger.info("distributed init (rank {}): {}".format(node_rank, ddp_url))
        
        dist_backend = getattr(opts, "ddp.backend", "nccl")

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                logger.log("Using NCCL as distributed backend with version={}".format(torch.cuda.nccl.version()))
            elif dist_backend is None:
                dist_backend = "gloo"

            dist.init_process_group(backend=dist_backend, init_method=ddp_url, world_size=world_size, rank=node_rank,)

            if torch.cuda.is_available():
                dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank