import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("警告: triton 不可用，相关功能将被跳过")
from execute_util import text, link, image
from file_util import ensure_directory_exists
from lecture_util import article_link
from torch_util import get_device
from lecture_06_utils import check_equal, check_equal2, get_local_url, round1, mean
import os

def main():
    announcements()

    text("上次课程：GPU 和性能的高层次概览")
    text("本次课程：基准测试/性能分析 + 编写 kernel")

    if not torch.cuda.is_available():
        text("你应该在 GPU 上运行本课程以获得完整体验。")

    review_of_gpus()
    benchmarking_and_profiling()  # Important for understanding!

    kernel_fusion_motivation()
    cuda_kernels()  # Write kernels in CUDA/C++
    if TRITON_AVAILABLE:
        triton_kernels()  # Write kernels in Python
    else:
        text("跳过 Triton kernels（Triton 在 macOS ARM64 上不可用）")
    pytorch_compilation()  # Don't write kernels at all?

    # More advanced computations
    if TRITON_AVAILABLE:
        triton_softmax_main()
    else:
        text("跳过 Triton softmax 示例（Triton 在 macOS ARM64 上不可用）")

    text("## 总结")

    text("编程模型（PyTorch、Triton、PTX）与硬件之间的差距 => 性能之谜")

    text("基准测试用于理解扩展性")
    text("性能分析用于理解 PyTorch 函数的内部实现（最终归结为 kernel）")
    text("查看 PTX 汇编以理解 CUDA kernel 的内部实现")

    text("编写函数的 5 种方式：手动、PyTorch、编译、CUDA、Triton")
    text("GeLU（逐元素）、softmax（逐行）、matmul（复杂聚合）")

    text("关键原则：组织计算以最小化读/写")
    text("关键思想：kernel 融合（仓库/工厂类比）、分块（共享内存）")
    text("自动编译器（Triton、torch.compile）将随着时间推移变得更好")

    further_reading()


def announcements():
    text("作业 1 排行榜 "), link(title="[Leaderboard]", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")
    text("作业 2 已发布 "), link(title="[A2]", url="https://github.com/stanford-cs336/spring2025-assignment2-systems")


def review_of_gpus():
    text("## 硬件")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=800)
    text("计算：流式多处理器 (SMs) [A100: 108]")
    text("内存：")
    text("- DRAM [A100: 80GB] - 大、慢")
    text("- L2 cache [A100: 40MB]")
    text("- L1 cache [A100: 192KB per SM] - 小、快")

    text("你可以查看实际 GPU 的规格。")
    print_gpu_specs()

    text("基本结构：对所有 i = 0, ..., N-1 运行 f(i)")

    text("## 执行模型")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    text("- *Thread*：处理单个索引（即 f(i)）")
    text("- *Thread block*（又称并发线程数组）：调度到单个 SM 上")
    text("- *Grid*：thread block 的集合")

    text("为什么需要 thread block？共享内存。")
    text("- 直觉：将读取相似数据的 f(i) 分组在一起")
    text("- thread block 内的线程拥有共享内存（与 L1 cache 一样快）[A100: 164KB]")
    text("- 可以在 block 内同步线程（用于读/写）（但不能跨 block）")

    text("### 硬件和执行的交互")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    text("Thread block 以波次方式调度到 SM 上。")
    text("问题：最后一波的 thread block 较少，导致一些 SM 空闲（低占用率）。")
    text("波次量化：使 thread block 数量能被 SM 数量整除。")
    text("经验法则：thread block 数量应该 >= 4x SM 数量")
    text("挑战：硬件的某些方面对执行模型是隐藏的（例如调度、SM 数量）。")

    text("### 算术强度：# FLOPs / # bytes")
    text("- 如果高，操作是计算受限的（好）")
    text("- 如果低，操作是内存受限的（坏）")
    text("通用规则：矩阵乘法是计算受限的，其他所有操作都是内存受限的")


def benchmarking_and_profiling():
    text("重要：对你的代码进行基准测试/性能分析！")

    text("你可以阅读规格表（营销材料）和论文")
    text("...但性能取决于你的库版本、你的硬件、你的工作负载")
    text("...所以对代码进行基准测试/性能分析是无可替代的。")

    text("示例计算：在 MLP 上运行前向/反向传播。")
    run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=5)

    benchmarking()       # 需要多长时间？
    profiling()          # 时间花在哪里？

    text("每次做出更改时，都要进行基准测试/性能分析！")


class MLP(nn.Module):
    """简单的 MLP：linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # 定义一个模型（随机权重）
    model = MLP(dim, num_layers).to(get_device())

    # 定义一个输入（随机）
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # 运行模型 `num_steps` 次（注意：没有优化器更新）
        for step in range(num_steps):
            # 前向传播
            y = model(x).mean()

            # 反向传播
            y.backward()

    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # 设置：创建一个随机的 dim x dim 矩阵
    x = torch.randn(dim, dim, device=get_device())
    # 返回一个执行操作的函数
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # 设置：创建两个随机的 dim x dim 矩阵
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # 返回一个执行操作的函数
    return lambda : operation(x, y)


def benchmarking():
    text("基准测试测量执行某个操作的实际时间。")

    text("它只给出端到端的时间，而不是时间花在哪里（性能分析）。")

    text("它仍然有用于：")
    text("- 比较不同的实现（哪个更快？），以及")
    text("- 理解性能如何扩展（例如，随维度变化）。")

    text("让我们定义一个方便的函数来对任意函数进行基准测试。")
    benchmark("sleep", lambda : time.sleep(50 / 1000))

    text("### 基准测试矩阵乘法")
    text("首先，让我们对方阵的矩阵乘法进行基准测试。")
    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)  # @inspect dims
    else:
        dims = (1024, 2048)  # @inspect dims
    
    matmul_results = [] 
    for dim in dims:
        # @ inspect dim
        result = benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
        matmul_results.append((dim, result))  # @inspect matmul_results

    text("让我们对 MLP 进行基准测试！")
    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers 
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps

    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)) # @inspect mlp_base


    text("扩展步数。")
    step_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=scale * num_steps)) # @inspect result, @inspect scale, @inspect num_steps
        step_results.append((scale, result))  # @inspect step_results

    text("扩展层数。")
    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)", 
                         run_mlp(dim=dim, num_layers=scale * num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        layer_results.append((scale, result))  # @inspect layer_results

    text("扩展批次大小。")
    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=scale * batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        batch_results.append((scale, result))  # @inspect batch_results

    text("扩展维度。")
    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)", 
                         run_mlp(dim=scale * dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        dim_results.append((scale, result))  # @inspect dim_results

    text("由于 CUDA kernel、硬件等的非均质性，时间并不总是可预测的。")

    text("你也可以使用 `torch.utils.benchmark`，它提供了更多便利。"), 
    link("https://pytorch.org/tutorials/recipes/recipes/benchmark.html")
    text("我们没有使用它是为了使基准测试更加透明。")


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """通过运行 `num_trials` 次来对 `func` 进行基准测试，并返回所有时间。"""
    # 预热：由于编译、缓存等原因，第一次可能会较慢。
    # 由于我们会多次运行 kernel，重要的是稳态时间。
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA 线程完成（重要！）

    # 现在真正计时！
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # 多次执行以捕获方差
        start_time = time.time()

        run()  # 实际执行计算
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待 CUDA 线程完成（重要！）

        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    return mean_time


def profiling():
    text("基准测试关注端到端时间，而性能分析关注时间花在哪里。")
    text("显而易见：性能分析帮助你理解时间花在哪里。")
    text("更深层次：性能分析帮助你理解（调用了什么）。")

    text("PyTorch 有一个很好的内置性能分析器 "), link("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    text("让我们对一些代码进行性能分析，看看底层发生了什么。")
    sleep_function = lambda : time.sleep(50 / 1000)
    sleep_profile = profile("sleep", sleep_function) 
    text(f"## sleep")
    text(sleep_profile, verbatim=True)
    

    text("让我们从一些基本操作开始。")
    add_function = lambda a, b: a + b
    add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
    text(f"## add")
    text(add_profile, verbatim=True)

    matmul_function = lambda a, b: a @ b
    matmul_profile = profile("matmul", run_operation2(dim=2048, operation=matmul_function))
    text(f"## matmul")
    text(matmul_profile, verbatim=True)

    matmul_function_128 = lambda a, b: a @ b
    matmul_profile_128 = profile("matmul(dim=128)", run_operation2(dim=128, operation=matmul_function_128))
    text(f"## matmul(dim=128)")
    text(matmul_profile_128, verbatim=True)

    text("观察")
    text("- 你可以看到实际调用了哪些 CUDA kernel。")
    text("- 根据张量维度的不同，会调用不同的 CUDA kernel。")

    text("CUDA kernel 的名称告诉我们一些关于实现的信息。")
    text("示例：cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    text("- cutlass：NVIDIA 的线性代数 CUDA 库")
    text("- 256x128：分块大小")

    text("现在让我们看一些复合操作。")
    cdist_function = lambda a, b: torch.cdist(a, b)
    cdist_profile = profile("cdist", run_operation2(dim=2048, operation=cdist_function))
    text(f"## cdist")
    text(cdist_profile, verbatim=True)

    gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
    gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))
    text(f"## gelu")
    text(gelu_profile, verbatim=True)

    softmax_function = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
    softmax_profile = profile("softmax", run_operation2(dim=2048, operation=softmax_function))
    text(f"## softmax")
    text(softmax_profile, verbatim=True)

    text("现在让我们对 MLP 进行性能分析。")
    text("我们还将使用火焰图可视化堆栈跟踪，这揭示了时间花在哪里。")
    if torch.cuda.is_available():
        mlp_profile = profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
    else:
        mlp_profile = profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)
    text(f"## mlp")
    text(mlp_profile, verbatim=True)


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # 预热
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA 线程完成（重要！）

    # 使用性能分析器运行代码
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # 输出堆栈跟踪用于可视化
            with_stack=with_stack,
            # 导出堆栈跟踪用于可视化所需
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待 CUDA 线程完成（重要！）

    # 打印表格
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)

    # 写入堆栈跟踪可视化
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table

def kernel_fusion_motivation():
    text("Horace He 的博客文章 "), link(title="[Article]", url="https://horace.io/brrr_intro.html")

    text("类比：仓库 : DRAM :: 工厂 : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png", width=800)

    text("每个操作都需要读取/计算/写入：")
    image("https://horace.io/img/perf_intro/multi_operators.png", width=800)

    text("如果我们*融合*操作，只需要读/写一次：")
    image("https://horace.io/img/perf_intro/operator_fusion.png", width=800)

    text("为了看到融合的效果，让我们考虑 GeLU 激活函数。"), 
    link("https://pytorch.org/docs/stable/generated/torch.nn.GELU.html")

    text("让我们考虑两种计算 GeLU 的方法：")
    x = torch.tensor([1.])  # @inspect x

    text("1. 默认的 PyTorch 实现（融合的）：")
    y1 = pytorch_gelu(x)  # @inspect y1

    text("2. 我们也可以手动编写（未融合）：")
    y2 = manual_gelu(x)  # @inspect y2

    # 检查实现是否匹配
    assert torch.allclose(y1, y2)

    # 更系统地检查
    check_equal(pytorch_gelu, manual_gelu)

    text("让我们进行基准测试。")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    if manual_time is not None and pytorch_time is not None:
        text(f"融合版本明显更快：{manual_time:.2f} ms, {pytorch_time:.2f} ms")
    else:
        text("无法比较时间 - 基准测试结果为 None")

    text("让我们看看底层。")
    manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    text(f"## manual_gelu")
    text(manual_gelu_profile, verbatim=True)
    pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    text(f"## pytorch_gelu")
    text(pytorch_gelu_profile, verbatim=True)
    text("PyTorch 只调用一个 kernel，而其他的是原子的（记住仓库/工厂）")

    text(f"## 查看 MLP 的 Nsight 性能分析器   ")


def cuda_kernels():
    text("现在让我们打开盒子，通过编写自己的 CUDA kernel 来理解其内部发生了什么。")

    text("让我们用 CUDA 编写 GeLU 函数。")
    cuda_gelu = create_cuda_gelu() # @inspect cuda_gelu
    x = manual_gelu # @inspect x

    text("检查我们实现的正确性。")
    if cuda_gelu is not None:
        check_equal(cuda_gelu, manual_gelu)

    text("对我们的 CUDA 版本进行基准测试。")
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    if cuda_gelu is not None:
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu)) # @inspect cuda_time 
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
        text(f"## cuda_gelu")
        text(cuda_gelu_profile, verbatim=True)
    text("我们的 CUDA 实现比手动版本快，但不如 PyTorch。")

    text("逐元素操作在 CUDA 中很容易（尽管你仍然可以更聪明）。")
    text("但大多数有趣的操作（例如 matmul、softmax、RMSNorm）需要读取多个值。")
    text("为此，你必须考虑管理共享内存等。")


def create_cuda_gelu():
    text("CUDA 是 C/C++ 的扩展，带有用于管理 GPU 的 API。")

    text("简化图景：编写 f(i)，CUDA kernel 为所有 i 计算 f(i)。")

    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=0.5)
    text("Grid：thread block 的集合：numBlocks = (2, 4), blockDim = (1, 8)")
    text("Thread block：线程的集合：blockIdx = (0, 1)")
    text("Thread：单个操作单元：threadIdx = (0, 3)。")

    text("你编写线程执行的代码，使用 (blockIdx, blockDim, threadIdx) 来确定要做什么。")

    text("设置 CUDA_LAUNCH_BLOCKING，这样如果有错误，CUDA 会告诉你出了什么问题。")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    text("`load_inline` 函数使编写 CUDA 代码并将其绑定到 Python 模块以供立即使用变得方便。")

    # CUDA 代码：包含完整逻辑
    cuda_gelu_src = open("gelu.cu").read()
    text(cuda_gelu_src, verbatim=True)

    # C++ 代码：定义 gelu 函数
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    text("编译 CUDA 代码并将其绑定到 Python 模块。")
    ensure_directory_exists("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )

    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu


def triton_kernels():
    triton_introduction()
    triton_gelu_main()


def triton_introduction():
    text("由 OpenAI 于 2021 年开发 "), 
    link("https://openai.com/research/triton")

    text("使 GPU 编程更易于访问")
    text("- 用 Python 编写")
    text("- 考虑 thread block 而不是线程")

    text("Triton 提供了什么？", verbatim=True)
    text("                                             CUDA      Triton", verbatim=True)
    text("- Memory coalescing (从 DRAM 传输)          manual    automatic", verbatim=True)
    text("- 共享内存管理                               manual    automatic", verbatim=True)
    text("- SM 内调度                                  manual    automatic", verbatim=True)
    text("- 跨 SM 调度                                 manual    manual", verbatim=True)

    text("编译器做了更多工作，实际上可以超越 PyTorch 实现！")


def triton_gelu_main():
    if not torch.cuda.is_available():
        return

    text("Triton 的一个巨大优势是你可以逐步执行 Python 代码。")

    text("让我们逐步执行一个 Triton kernel。")
    x = torch.randn(8192, device=get_device())
    y1 = triton_gelu(x)

    print_ptx_main()  # 查看生成的指令

    text("检查它是否正确。")
    check_equal(triton_gelu, manual_gelu)

    text("现在让我们将它与 PyTorch 和 CUDA 实现进行基准测试比较。")
    text("记住设置 TRITON_INTERPRET=0 以获得良好性能。")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time

    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    text(f"## triton_gelu")
    text(triton_gelu_profile, verbatim=True)

    text("我们的 Triton 实现（triton_gelu）：")
    text("- 几乎与 PyTorch 实现（pytorch_gelu）一样好。")
    text("- 实际上比我们的朴素 CUDA 实现（cuda_gelu）慢。")

    text("Triton 操作 block，CUDA 操作线程。")
    text("Block 允许 Triton 编译器进行其他优化（例如线程粗化）。")

    text("一切都比手动实现（manual_gelu）快得多。")


def triton_gelu(x: torch.Tensor):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton 不可用")
    assert x.is_cuda
    assert x.is_contiguous()

    # 分配输出张量
    y = torch.empty_like(x)

    # 确定网格（元素分成 block）
    num_elements = x.numel()
    block_size = 1024  # 线程数
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


if TRITON_AVAILABLE:
    @triton.jit
    def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
        # 输入在 `x_ptr`，输出在 `y_ptr`
        #     |        Block 0            |          Block 1          |      ...      |
        #                            BLOCK_SIZE                                 num_elements

        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE

        # 此 thread block 应该操作的索引
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # 处理边界
        mask = offsets < num_elements

        # 读取
        x = tl.load(x_ptr + offsets, mask=mask)

        # 近似 gelu 是 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # 计算（tl.tanh 不存在，使用 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
        a = 0.79788456 * (x + 0.044715 * x * x * x)
        exp = tl.exp(2 * a)
        tanh = (exp - 1) / (exp + 1)
        y = 0.5 * x * (1 + tanh)

        # 存储
        tl.store(y_ptr + offsets, y, mask=mask)
else:
    def triton_gelu_kernel(*args, **kwargs):
        raise RuntimeError("Triton 不可用")


def print_ptx_main():
    text("PTX（并行线程执行）就像 GPU 的汇编语言。")

    text("我们可以看到 Triton 生成的 PTX 代码。")
    link("https://docs.nvidia.com/cuda/parallel-thread-execution/index.html")

    ptx = print_ptx("triton_gelu", triton_gelu_kernel)
    text(ptx, verbatim=True)

    text("观察：")
    text("- ld.global.* 和 st.global.* 从全局内存读取和写入")
    text("- %ctaid.x 是 block 索引，%tid.x 是线程索引")
    text("- %f* 是浮点寄存器，%r* 是整数寄存器")
    text("- 一个线程同时处理 8 个元素（线程粗化）")

    
def print_ptx(name: str, kernel):
    if not TRITON_AVAILABLE:
        text("Triton 不可用，跳过 PTX 生成。")
        return ""
    
    if os.environ.get("TRITON_INTERPRET") == "1":
        text("在解释模式下不生成 PTX。")
        return

    """打印 Triton 为给定 `kernel` 生成的 PTX 代码。"""
    ptx_path = f"var/{name}-ptx.txt"
    text("让我们查看 PTX 代码。")
    link(get_local_url(ptx_path))

    with open(ptx_path, "w") as f:
        return list(kernel.cache[0].values())[0].asm["ptx"]

    


def pytorch_compilation():
    text("到目前为止，我们已经看到了编写 GeLU 的三种方法：")
    text("- 使用默认的 PyTorch 函数")
    text("- 用 Python 编写 "), link(manual_gelu)
    text("- 用 CUDA 编写 "), link(create_cuda_gelu)
    text("- 用 Triton 编写 "), link(triton_gelu)

    text("- 用 Python 编写并编译成 Triton")
    compiled_gelu = torch.compile(manual_gelu)

    text("检查我们实现的正确性。")
    check_equal(compiled_gelu, manual_gelu)

    if not torch.cuda.is_available():
        return

    text("让我们进行基准测试和性能分析！")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time
    compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time

    text("让我们看看底层")
    compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    text(f"## compiled_gelu")
    text(compiled_gelu_profile, verbatim=True)


def triton_softmax_main():
    text("到目前为止，我们已经看过 Triton 中的逐元素操作（例如 GeLU）。")
    text("现在让我们看看对多个值进行聚合的操作。")

    text("我们将大致遵循 Triton 融合 softmax 教程："), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    text("回想一下，softmax 操作用于 attention 和生成概率。")
    text("归一化矩阵的每一行：")
    text("[A1 A2 A3]   =>   [A1/A A2/A A3/A]", verbatim=True)
    text("[B1 B2 B3]   =>   [B1/B B2/B B3/B]", verbatim=True)

    text("让我们首先从朴素实现开始，并跟踪读/写。")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=get_device())
    y1 = manual_softmax(x) # @inspect y1

    if not torch.cuda.is_available():
        return

    text("现在让我们编写 Triton kernel。")
    y2 = triton_softmax(x)
    assert torch.allclose(y1, y2)

    text("检查我们的实现是否正确。")
    check_equal2(pytorch_softmax, manual_softmax)
    check_equal2(pytorch_softmax, triton_softmax)

    compiled_softmax = torch.compile(manual_softmax)

    text("现在让我们对所有内容进行基准测试。")
    manual_time = benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax)) # @inspect manual_time
    compiled_time = benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax)) # @inspect compiled_time
    pytorch_time = benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax)) # @inspect pytorch_time
    triton_time = benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax)) # @inspect triton_time

    text("使用性能分析器查看底层。")
    manual_softmax_profile = profile("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
    text(f"## manual_softmax")
    text(manual_softmax_profile, verbatim=True)
    compiled_softmax_profile = profile("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
    text(f"## compiled_softmax")
    text(compiled_softmax_profile, verbatim=True)
    pytorch_softmax_profile = profile("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
    text(f"## pytorch_softmax")
    text(pytorch_softmax_profile, verbatim=True)
    triton_softmax_profile = profile("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))
    text(f"## triton_softmax")
    text(triton_softmax_profile, verbatim=True)

    text("最后让我们看看 PTX 代码。")
    ptx = print_ptx("triton_softmax", triton_softmax_kernel)
    text(ptx, verbatim=True)


def manual_softmax(x: torch.Tensor):
    # M：行数，N：列数
    M, N = x.shape

    # 计算每行的最大值（MN 次读取，M 次写入）
    x_max = x.max(dim=1)[0]

    # 减去最大值（MN + M 次读取，MN 次写入）
    x = x - x_max[:, None]

    # 指数化（MN 次读取，MN 次写入）
    numerator = torch.exp(x)

    # 计算归一化常数（MN 次读取，M 次写入）
    denominator = numerator.sum(dim=1)

    # 归一化（MN 次读取，MN 次写入）
    y = numerator / denominator[:, None]

    # 总计：5MN + M 次读取，3MN + 2M 次写入
    # 原则上应该有 MN 次读取，MN 次写入（4 倍加速！）
    return y


def triton_softmax(x: torch.Tensor):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton 不可用")
    # 分配输出张量
    y = torch.empty_like(x)

    # 确定网格
    M, N = x.shape                          # 行数 x 列数
    block_size = triton.next_power_of_2(N)  # 每个 block 包含所有列
    num_blocks = M                          # 每个 block 是一行

    # 启动 kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


if TRITON_AVAILABLE:
    @triton.jit
    def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
        assert num_cols <= BLOCK_SIZE

        # 独立处理每一行
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # 从全局内存读取
        x_start_ptr = x_ptr + row_idx * x_row_stride
        x_ptrs = x_start_ptr + col_offsets
        x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

        # 计算
        x_row = x_row - tl.max(x_row, axis=0)
        numerator = tl.exp(x_row)
        denominator = tl.sum(numerator, axis=0)
        y_row = numerator / denominator

        # 写回全局内存
        y_start_ptr = y_ptr + row_idx * y_row_stride
        y_ptrs = y_start_ptr + col_offsets
        tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
else:
    def triton_softmax_kernel(*args, **kwargs):
        raise RuntimeError("Triton 不可用")


def triton_matmul_main():
    text("矩阵乘法可能是有史以来优化最多的算法。")

    text("如果你用 CUDA 编写矩阵乘法，你必须做各种疯狂的事情。")
    link("https://github.com/openai/blocksparse/blob/master/src/matmul_op_gpu.cu")

    text("在 Triton 中要容易得多。")
    link("https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html")

    text("       k                  j                     ", verbatim=True)
    text("  [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    text("i [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    text("  [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    text("朴素地：需要 MKN 次读取，MN 次写入")

    text("计算 C4 和 C5 都需要 A4、A5、A6。")
    text("我们能从 DRAM 读取一次 A4、A5、A6 来计算两者吗？")
    text("答案：可以，使用共享内存！")

    text("## 分块（利用共享内存）")

    text("回想一下共享内存是：")
    text("- 快（快 10 倍）且小（~100KB）")
    text("- 在 block 中的所有线程之间共享。")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg")

    text("简单情况：对于小矩阵，将 A 和 B 的全部加载到共享内存中，然后可以计算 C。")
    text("现在我们得到 MK + KN 次读取，MN 次写入")

    text("但如果我们有大矩阵...")

    image("https://www.researchgate.net/profile/Axel-Huebl/publication/320499173/figure/fig1/AS:614298980196359@1523471698396/Performance-critical-A-B-part-of-the-GEMM-using-a-tiling-strategy-A-thread-iterates.png", width=0.5)
    text("关键思想：将矩阵分成块。")
    text("对于 A 的每个块和 B 的每个块：")
    text("- 加载到共享内存，")
    text("- 做小矩阵乘法，")
    text("- 写入部分和。")

    text("分块矩阵乘法的动画 "), link("https://youtu.be/aMvCEEBIBto")

    text("## 利用 L2 cache")

    text("计算矩阵 9 个元素的两种方法：")
    image("https://triton-lang.org/main/_images/grouped_vs_row_major_ordering.png", width=0.5)
    text("1. 加载 9 + 81 = 90 个块")
    text("1. 加载 27 + 27 = 54 个块")

    text("按最小化读取的顺序处理块。")

    text("为什么要为矩阵乘法编写自己的 kernel（例如 A @ B）？")
    text("答案：与另一个操作融合（例如 gelu(A @ B)）")

    if not torch.cuda.is_available():
        return
    text("让我们试试！")
    benchmark("pytorch_matmul", run_operation2(dim=16384, operation=torch.matmul))
    benchmark("triton_matmul", run_operation2(dim=16384, operation=triton_matmul))

    # 由于某种原因不工作
    #print_ptx("triton_matmul", triton_matmul_kernel)


def further_reading():
    text("Horace He 的博客文章 "), link(title="[Article]", url="https://horace.io/brrr_intro.html")

    text("CUDA MODE 讲座 1：如何在 PyTorch 中对 CUDA kernel 进行性能分析 "), link(title="[Video]", url="https://www.youtube.com/watch?v=LuhJEEJQgUM")
    text("CUDA MODE 讲座 2：PPMP 书的第 1-3 章 "), link(title="[Video]", url="https://www.youtube.com/watch?v=NQ-0D5Ti2dc")
    text("CUDA MODE 讲座 3：Python 程序员的 CUDA 入门 "), link(title="[Video]", url="https://www.youtube.com/watch?v=4sgKnKbR-WE")
    text("CUDA MODE 讲座 4：计算和内存基础 "), link(title="[Video]", url="https://www.youtube.com/watch?v=lTmYrKwjSOU")
    text("CUDA MODE 讲座 8：CUDA 性能检查清单 "), link(title="[Video]", url="https://www.youtube.com/watch?v=SGhfUhlowB4")

    text("HetSys 课程：讲座 1：使用 GPU 编程异构计算系统 "), link(title="[Video]", url="https://www.youtube.com/watch?v=8JGo2zylE80")
    text("HetSys 课程：讲座 2：SIMD 处理和 GPU "), link(title="[Video]", url="https://www.youtube.com/watch?v=x1MA4MtO4Tc")
    text("HetSys 课程：讲座 3：GPU 软件层次结构 "), link(title="[Video]", url="https://www.youtube.com/watch?v=KGZ00J5MJz0")
    text("HetSys 课程：讲座 4：GPU 内存层次结构 "), link(title="[Video]", url="https://www.youtube.com/watch?v=ZQKMZIP3Fzg")
    text("HetSys 课程：讲座 5：GPU 性能考虑 "), link(title="[Video]", url="https://www.youtube.com/watch?v=ODeprwr3Jho")

    link(title="[A100 GPU with NVIDIA Ampere Architecture]", url="https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e")
    link(title="[NVIDIA Deep Learning Performance Guide]", url="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")
    link(title="[GPU Puzzles]", url="https://github.com/srush/gpu-puzzles")
    link(title="[Triton Paper]", url="https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf")
    link(title="[PyTorch 2.0 Acceleration]", url="https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26")

############################################################

def print_gpu_specs():
    num_devices = torch.cuda.device_count()  # @inspect num_devices
    text(f"{num_devices} 个设备")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties
        text(f"{i}: {properties}")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def pytorch_gelu(x: torch.Tensor):
    # 使用 tanh 近似以匹配我们的实现
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))




if __name__ == "__main__":
    main()
