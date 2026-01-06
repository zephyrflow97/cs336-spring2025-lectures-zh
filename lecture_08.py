import torch
import time
import os
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
from execute_util import text, image, link, system_text
from torch_util import get_device
from lecture_util import article_link
from lecture_08_utils import spawn, int_divide, summarize_tensor, get_init_params, render_duration

def main():
    text("上周：单个 GPU 内的并行")
    text("本周：跨多个 GPU 的并行")
    image("images/gpu-node-overview.png", width=500)

    text("在这两种情况下，**计算**（算术逻辑单元）都远离输入/输出（**数据**）。")
    text("统一主题：编排计算以避免数据传输瓶颈")

    text("上周：通过融合/分块减少内存访问")
    text("本周：通过复制/分片减少跨 GPU/节点的通信")

    text("广义层次结构（从小/快到大/慢）：")
    text("- 单节点，单 GPU：L1 cache / 共享内存")
    text("- 单节点，单 GPU：HBM")
    text("- 单节点，多 GPU：NVLink")
    text("- 多节点，多 GPU：NVSwitch")

    text("本次课程：用代码具体化上次课程的概念")

    link(title="[stdout for this lecture]", url="var/traces/lecture_08_stdout.txt")

    text("### 第 1 部分：分布式通信/计算的构建块")
    collective_operations()    # 概念性编程接口
    torch_distributed()        # 这在 NCCL/PyTorch 中如何实现
    benchmarking()             # 测量实际 NCCL 带宽

    text("### 第 2 部分：分布式训练")
    text("在深度 MLP 上演示每种策略的基本实现。")
    text("回想一下，MLP 是 Transformer 中的计算瓶颈，所以这具有代表性。")
    data_parallelism()         # 沿批次维度切分
    tensor_parallelism()       # 沿宽度维度切分
    pipeline_parallelism()     # 沿深度维度切分

    text("缺少什么？")
    text("- 更通用的模型（带 attention 等）")
    text("- 更多通信/计算重叠")
    text("- 这需要更复杂的代码和更多记账工作")
    text("- Jax/TPUs：只需定义模型、分片策略，Jax 编译器处理其余部分 "), link(title="[levanter]", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    text("- 但我们使用 PyTorch，这样你可以看到如何从原语构建")

    text("### 总结")
    text("- 并行化的多种方式：data（批次）、tensor/expert（宽度）、pipeline（深度）、sequence（长度）")
    text("- 可以**重新计算**或存储在**内存**中或存储在另一个 GPU 的内存中并**通信**")
    text("- 硬件越来越快，但总是想要更大的模型，所以会有这种层次结构")


def collective_operations():
    text("**集合操作**是用于分布式编程的概念性原语 "), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    text("- 集合意味着你指定跨多个（例如 256 个）节点的通信模式。")
    text("- 这些在 1980 年代的并行编程文献中就很经典。")
    text("- 比自己管理点对点通信更好/更快的抽象。")

    text("术语：")
    text("- **World size**：设备数量（例如 4）")
    text("- **Rank**：一个设备（例如 0、1、2、3）")

    text("### Broadcast"), image("https://pytorch.org/tutorials/_images/broadcast.png", width=400)

    text("### Scatter"), image("https://pytorch.org/tutorials/_images/scatter.png", width=400)

    text("### Gather"), image("https://pytorch.org/tutorials/_images/gather.png", width=400)

    text("### Reduce"), image("https://pytorch.org/tutorials/_images/reduce.png", width=400)

    text("### All-gather"), image("https://pytorch.org/tutorials/_images/all_gather.png", width=400)

    text("### Reduce-scatter"), image("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png", width=400)

    text("### All-reduce = reduce-scatter + all-gather"), image("https://pytorch.org/tutorials/_images/all_reduce.png", width=400)

    text("记住术语的方法：")
    text("- Reduce：执行某些结合/交换操作（sum、min、max）")
    text("- Broadcast/scatter 是 gather 的逆操作")
    text("- All：表示目的地是所有设备")


def torch_distributed():
    text("### 硬件")
    text("经典（家庭）：")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=400)
    text("- 同一节点上的 GPU 通过 PCI(e) 总线通信（v7.0，16 通道 => 242 GB/s）"), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    text("- 不同节点上的 GPU 通过以太网通信（~200 MB/s）")

    text("现代（数据中心）：")
    image("https://www.nextplatform.com/wp-content/uploads/2018/04/nvidia-nvswitch-topology-two.jpg", width=400)
    text("- 节点内：NVLink 直接连接 GPU，绕过 CPU")
    text("- 跨节点：NVSwitch 直接连接 GPU，绕过以太网")

    text("每个 H100 有 18 个 NVLink 4.0 链接，总共 900GB/s "), article_link("https://www.nvidia.com/en-us/data-center/nvlink/")
    text("相比之下，HBM 的内存带宽为 3.9 TB/s "), article_link("https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")

    text("让我们检查一下我们的硬件设置。"), article_link("https://guide.ncloud-docs.com/docs/en/server-baremetal-a100-check-vpc")
    if torch.cuda.is_available():
        os.system("nvidia-smi topo -m")
        text("注意 GPU 通过 NV18 连接，也连接到 NIC（用于 PCIe）")

    text("### NVIDIA Collective Communication Library (NCCL)")
    text("NCCL 将集合操作转换为在 GPU 之间发送的低级数据包。"), link(title="[talk]", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- 检测硬件拓扑（例如节点数、交换机、NVLink/PCIe）")
    text("- 优化 GPU 之间的路径")
    text("- 启动 CUDA kernel 以发送/接收数据")

    text("### PyTorch 分布式库（`torch.distributed`）")
    link(title="[Documentation]", url="https://pytorch.org/docs/stable/distributed.html")

    text("- 为集合操作提供清晰的接口（例如 `all_gather_into_tensor`）")
    text("- 支持不同硬件的多个后端：gloo（CPU）、nccl（GPU）")
    text("- 还支持更高级的算法（例如 `FullyShardedDataParallel`）[本课程未使用]")

    text("让我们演示一些示例。")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):
    """此函数为每个进程异步运行（rank = 0, ..., world_size - 1）。"""
    setup(rank, world_size)

    # All-reduce
    dist.barrier()  # 等待所有进程到达此点（在这种情况下，用于 print 语句）

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # 既是输入也是输出

    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # 就地修改张量
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # 输入
    output = torch.empty(1, device=get_device(rank))  # 分配输出

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # All-gather
    dist.barrier()

    input = output  # 输入是 reduce-scatter 的输出
    output = torch.empty(world_size, device=get_device(rank))  # 分配输出

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("确实，all-reduce = reduce-scatter + all-gather！")

    cleanup()


def benchmarking():
    text("让我们看看通信有多快（限制在一个节点）。")

    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    # 参考资料
    link(title="How to reason about operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="Sample code", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 创建张量
    tensor = torch.randn(num_elements, device=get_device(rank))

    # 预热
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA kernel 完成
        dist.barrier()            # 等待所有进程到达此处

    # 执行 all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA kernel 完成
        dist.barrier()            # 等待所有进程到达此处
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 测量有效带宽
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x 因为发送输入和接收输出
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 创建输入和输出
    input = torch.randn(world_size, num_elements, device=get_device(rank))  # 每个 rank 有一个矩阵
    output = torch.empty(num_elements, device=get_device(rank))

    # 预热
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA kernel 完成
        dist.barrier()            # 等待所有进程到达此处

    # 执行 reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA kernel 完成
        dist.barrier()            # 等待所有进程到达此处
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 测量有效带宽
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # 输入中有多少数据
    sent_bytes = data_bytes * (world_size - 1)  # 需要发送多少（这里没有 2x）
    total_duration = world_size * duration  # 传输的总时间
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def data_parallelism():
    image("images/data-parallelism.png", width=300)
    text("分片策略：每个 rank 获取数据的一个切片")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("注意：")
    text("- 各 rank 的损失不同（在本地数据上计算）")
    text("- 梯度通过 all-reduce 在各 rank 间保持相同")
    text("- 因此，参数在各 rank 间保持相同")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # 获取此 rank 的数据切片（实际上，每个 rank 应该只加载自己的数据）
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # 创建 MLP 参数 params[0], ..., params[num_layers - 1]（每个 rank 拥有所有参数）
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # 每个 rank 有自己的优化器状态

    for step in range(num_steps):
        # 前向传播
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # 损失函数是平均平方幅度

        # 反向传播
        loss.backward()

        # 在 worker 间同步梯度（标准训练和 DDP 之间的唯一区别）
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 更新参数
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("分片策略：每个 rank 获取每层的一部分，传输所有数据/激活")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # 分片 `num_dim`  @inspect local_num_dim

    # 创建模型（每个 rank 获取 1/world_size 的参数）
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # 前向传播
    x = data
    for i in range(num_layers):
        # 计算激活（batch_size x local_num_dim）
        x = x @ params[i]  # 注意：这只在参数的一个切片上
        x = F.gelu(x)

        # 为激活分配内存（world_size x batch_size x local_num_dim）
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # 通过 all gather 发送激活
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # 连接它们以获得 batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # 反向传播：作业练习

    cleanup()


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("分片策略：每个 rank 获取层的子集，传输所有数据/激活")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # 使用所有数据
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # 拆分层
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers

    # 每个 rank 获取层的子集
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # 前向传播

    # 分解为微批次以最小化气泡
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # 数据
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 为激活分配内存
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # 从前一个 rank 获取激活
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 计算分配给此 rank 的层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 发送到下一个 rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    text("未处理：重叠通信/计算以消除流水线气泡")

    # 反向传播：作业练习

    cleanup()

############################################################

def setup(rank: int, world_size: int):
    # 指定 master 所在位置（rank 0），用于协调（实际数据通过 NCCL 传输）
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
