from sympy import symbols, oo
from execute_util import text, link, image
from lecture_util import article_link
from references import Reference, llama3, gqa, mla, longformer, sparse_transformer, mistral_7b

# 定义对应于 Transformer 模型形状的符号
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # 只是一个帮助取极限的常数
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    text("**推理**：给定一个**固定模型**，根据提示生成响应")

    text("### 理解推理工作负载")
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### 采取捷径（有损）")
    reduce_kv_cache_size()
    alternatives_to_the_transformer()
    quantization()
    model_pruning()

    text("总结：在不损害准确性的情况下降低推理复杂度")

    text("从头开始的方法：")
    text("1. 定义更快的模型架构")
    text("2. 训练更快的模型")

    text("蒸馏方法：")
    text("1. 定义更快的模型架构")
    text("2. 使用原始模型初始化权重（具有不同的架构）")
    text("3. 修复更快的模型（蒸馏）")

    text("### 使用捷径但要仔细检查（无损）")
    speculative_sampling()

    text("### 处理动态工作负载")
    text("在实时流量中对序列进行批处理很棘手，因为：")
    text("1. 请求在不同时间到达（等待批次对早期请求不利）")
    text("2. 序列有共享前缀（例如系统提示、生成多个样本）")
    text("3. 序列有不同长度（填充效率低）")

    continuous_batching()
    paged_attention()

    text("### 总结")
    text("- 推理很重要（实际使用、评估、强化学习）")
    text("- 与训练相比具有不同特征（内存受限、动态）")
    text("- 技术：新架构、量化、剪枝/蒸馏、推测解码")
    text("- 来自系统的想法（推测执行、分页）")
    text("- 新架构有巨大的改进潜力")


def landscape():
    text("推理出现在许多地方：")
    text("- 实际使用（聊天机器人、代码补全、批量数据处理）")
    text("- 模型评估（例如指令遵循）")
    text("- 测试时计算（思考需要更多推理）")
    text("- 通过强化学习训练（样本生成，然后评分）")

    text("为什么**效率**很重要：训练是一次性成本，推理会重复多次")
    image("images/openai-100b-tokens.png", width=600); link(title=" [tweet]", url="https://x.com/sama/status/1756089361609981993")
    image("images/cursor-1b-lines.png", width=600); link(title=" [tweet]", url="https://x.com/amanrsanger/status/1916968123535880684")

    text("指标：")
    text("- Time-to-first-token (TTFT)：用户在任何生成发生之前等待的时间（对交互式应用很重要）")
    text("- 延迟（秒/token）：token 对用户显示的速度（对交互式应用很重要）")
    text("- 吞吐量（token/秒）：对批处理应用有用")

    text("效率的关键考虑因素：")
    text("- 训练（监督）：你看到所有 token，可以在序列上并行化（Transformer 中的 matmul）")
    text("- 推理：你必须顺序生成，不能并行化，所以更难充分利用计算")

    text("进行推理的公司（对任何有产品或平台的人来说都是大事）：")
    text("- 提供封闭模型的提供商（OpenAI、Anthropic、Google 等）")
    text("- 提供开放权重模型的提供商（Together、Fireworks、DeepInfra 等）")

    text("开源包：")
    text("- vLLM (Berkeley) "), link(title="[talk]", url="https://www.youtube.com/watch?v=8BaEwoTk8XI")
    text("- Tensor-RT (NVIDIA) "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- TGI (Hugging Face) "), article_link("https://huggingface.co/docs/text-generation-inference/en/index")


def review_transformer():
    link(scaling_book_transformers)
    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    text("简化（遵循惯例）：`F = 4*D, D = N*H, N = K*G, S = T`")
    text("前向传播的 FLOPs：6 * (B*T) * (num_params + O(T))")


def review_of_arithmetic_intensity():
    text("设置：矩阵乘法 X (B x D) 和 W (D x F)")
    text("直觉：B 是批次大小，D 是隐藏维度，F 是 MLP 中的上投影维度")

    text("让我们对矩阵乘法 (X * W) 进行 FLOPs 和内存读写计算。")
    flops = 0
    bytes_transferred = 0

    text("步骤：")
    text("1. 从 HBM 读取 X (B x D)")
    bytes_transferred += 2*B*D
    text("2. 从 HBM 读取 W (D x F)")
    bytes_transferred += 2*D*F
    text("3. 计算 Y = X (B x D) @ W (D x F)")
    flops += 2*B*D*F
    text("4. 将 Y (B x F) 写入 HBM")
    bytes_transferred += 2*B*F

    text("让我们总结一下计算结果。")
    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F
    text("回顾一下，**算术强度**是指每传输一个字节我们执行多少计算（希望越高越好）。")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    text("假设 B 远小于 D 和 F，那么我们可以简化：")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("H100 的加速器强度：")
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    text("如果计算强度 > 加速器强度，则**计算受限**（好）")
    text("如果计算强度 < 加速器强度，则**内存受限**（坏）")
    text("结论：当且仅当 B > 295 时为计算受限")

    text("极端情况（B = 1，对应于矩阵-向量乘积）：")
    text("- 算术强度：1")
    text("- 内存受限（读取 D x F 矩阵，仅执行 2*D*F FLOPs）")
    text("- 这基本上就是生成时发生的情况...")


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    text("朴素推理：为了生成每个 token，将历史输入 Transformer")
    text("复杂度：生成 T 个 token 需要 O(T^3) FLOPs（一次前向传播是 O(T^2)）")

    text("观察：很多工作可以在前缀之间共享")
    text("解决方案：在 HBM 中存储 **KV cache**")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    text("KV cache：对于每个序列 (B)、token (S)、层 (L)、头 (K)，存储一个 H 维向量")

    text("推理的两个阶段：")
    text("1. **Prefill**：给定提示，编码为向量（可以像训练一样并行化）")
    text("2. **Generation**：生成新的响应 token（顺序）")

    text("让我们计算 MLP 和 attention 层的 FLOPs 和内存 IO。")
    text("S 是我们条件化的 token 数量，T 是我们生成的 token 数量。")
    text("稍后，我们将专门处理 prefill (T = S) 和 generation (T = 1)。")

    text("### MLP 层（仅查看矩阵乘法）")
    flops = 0
    bytes_transferred = 0
    text("步骤：")
    text("1. 从 HBM 读取 X (B x T x D)")
    bytes_transferred += 2*B*T*D
    text("2. 从 HBM 读取 Wup (D x F), Wgate (D x F), Wdown (F x D)")
    bytes_transferred += 3 * 2*D*F
    text("3. 计算 U = X (B x T x D) @ Wup (D x F)")
    flops += 2*B*T*D*F
    text("4. 将 U (B x T x F) 写入 HBM")
    bytes_transferred += 2*B*T*F
    text("5. 计算 G = X (B x T x D) @ Wgate (D x F)")
    flops += 2*B*T*D*F
    text("6. 将 G (B x T x F) 写入 HBM")
    bytes_transferred += 2*B*T*F
    text("7. 计算 Y = GeLU(G)*U (B x T x F) @ Wdown (F x D)")
    flops += 2*B*T*D*F
    text("8. 将 Y (B x T x D) 写入 HBM")
    bytes_transferred += 2*B*T*D

    text("让我们总结一下计算结果。")
    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("假设 B*T 远小于 D 和 F。")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("对于两个阶段：")
    text("1. Prefill：通过使 B T 足够大，很容易实现计算受限（好）")
    text("2. Generation：")
    text("- 一次生成一个 token (T = 1)")
    text("- B 是并发请求的数量，很难使其足够大！")

    text("### Attention 层（关注使用 FlashAttention 的矩阵乘法）")
    flops = 0
    bytes_transferred = 0
    text("步骤：")
    text("1. 从 HBM 读取 Q (B x T x D), K (B x S x D), V (B x S x D)")
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    text("2. 计算 A = Q (B x T x D) @ K (B x S x D)")
    flops += 2*B*S*T*D
    text("3. 计算 Y = softmax(A) (B x S x T x K x G) @ V (B x S x K x H)")
    flops += 2*B*S*T*D
    text("4. 将 Y (B x T x D) 写入 HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    text("对于两个阶段：")
    text("1. Prefill: T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # 好！
    text("2. Generation: T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # 坏！

    text("与 MLP 不同，不依赖于 B，所以批处理没有帮助！")
    text("为什么？")
    text("- 在 MLP 层中，每个序列都使用相同的 MLP 权重（Wup, Wgate, Wdown 不依赖于 B）")
    text("- 在 attention 层中，每个序列都有自己的向量 KV cache（Q, K, V 都依赖于 B）")

    text("总结")
    text("- Prefill 是计算受限的，generation 是内存受限的")
    text("- MLP 强度是 B（需要并发请求），attention 强度是 1（无法改进）")


def compute_transformer_stats(config):  # @inspect config
    """返回对应于 Transformer 各种统计信息的符号。"""
    text("内存、吞吐量和延迟取决于 Transformer 的形状。"), text(" "), link("")

    text("计算 Transformer 中的参数数量：")
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    text("要存储参数，只需使用 bf16（训练需要 fp32）")
    parameter_size = num_params * 2  # 2 表示 bf16
    
    text("我们也不需要梯度和优化器状态，因为我们不在训练。")
    text("但我们必须为每个序列（长度为 S）存储 KV cache（这是一些激活）：")
    text("每个序列需要存储多少：")
    kv_cache_size = S * (K*H) * L * 2 * 2  # 2 表示 key + value，2 表示 bf16

    text("总内存使用量：")
    memory = B * kv_cache_size + parameter_size
    text("延迟由内存 IO 决定（每步读取所有参数和 KV cache）")
    latency = memory / memory_bandwidth
    text("吞吐量是延迟的倒数，但我们并行生成 B 个 token")
    throughput = B / latency

    # 替换
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return num_params, memory, latency, throughput

def llama2_13b_config(args={}):
    return {S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000, memory_bandwidth: 3.35e12, **args}

def throughput_and_latency():
    text("因此我们已经证明推理是内存受限的。")
    text("现在让我们计算单个请求的理论最大延迟和吞吐量。")
    text("假设：可以完美地重叠计算和通信，并忽略各种类型的开销。")

    text("在 H100 上实例化 Llama 2 13B 的延迟和吞吐量：")
    config = llama2_13b_config()
    num_params, memory, latency, throughput = compute_transformer_stats(config)

    text("如果我们使用批次大小为 1：")
    bs1_memory = memory.subs(B, 1).simplify()   # @inspect bs1_memory
    bs1_latency = latency.subs(B, 1).simplify()   # @inspect bs1_latency
    bs1_throughput = throughput.subs(B, 1).simplify()   # @inspect bs1_throughput

    text("如果我们使用批次大小为 64（更差的延迟，更好的吞吐量）：")
    bs64_memory = memory.subs(B, 64).simplify()   # @inspect bs64_memory
    bs64_latency = latency.subs(B, 64).simplify()   # @inspect bs64_latency
    bs64_throughput = throughput.subs(B, 64).simplify()   # @inspect bs64_throughput

    text("如果我们使用批次大小为 256：")
    bs256_memory = memory.subs(B, 256).simplify()   # @inspect bs256_memory
    bs256_latency = latency.subs(B, 256).simplify()   # @inspect bs256_latency
    bs256_throughput = throughput.subs(B, 256).simplify()   # @inspect bs256_throughput
    text("不适合内存，但吞吐量增益也在递减...")

    text("延迟和吞吐量之间的**权衡**：")
    text("1. 较小的批次大小产生更好的延迟但更差的吞吐量")
    text("2. 较大的批次大小产生更好的吞吐量但更差的延迟")

    text("简单的并行化：如果你启动 M 个模型副本，延迟相同，吞吐量增加 M 倍！")
    text("更难的并行化：分片模型和 KV cache "), link(scaling_book_inference)

    text("注意：time-to-first-token (TTFT) 本质上是 prefill 的函数")
    text("在 prefill 期间使用较小的批次大小以获得更快的 TTFT")
    text("在 generation 期间使用较大的批次大小以提高吞吐量")


def reduce_kv_cache_size():
    text("回顾一下，内存是推理的瓶颈。")
    text("所以让我们尝试减少 KV cache 的大小")
    text("...但要确保我们不会损失太多准确性。")

    text("### Grouped-query attention (GQA) "), link(gqa)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    text("想法：N 个 query 头，但只有 K 个 key 和 value 头，每个与 N/K 个 query 头交互")
    text("Multi-headed attention (MHA): K=N")
    text("Multi-query attention (MQA): K=1")
    text("Group-query attention (GQA): K 介于两者之间")

    text("延迟/吞吐量改进：")
    image("images/gqa-speed.png", width=500); text(" "); link(gqa)
    text("将 KV cache 减少 N/K 倍")
    config = llama2_13b_config({K: 40, B: 64})  # 原始 Llama 2 13B
    k40_num_params, k40_memory, k40_latency, k40_throughput = compute_transformer_stats(config)  # @inspect k40_memory, @inspect k40_latency, @inspect k40_throughput

    config = llama2_13b_config({K: 8, B: 64})  # 使用 1:5 比例的 GQA
    k8_num_params, k8_memory, k8_latency, k8_throughput = compute_transformer_stats(config)  # @inspect k8_memory, @inspect k8_latency, @inspect k8_throughput

    text("这也意味着我们可以使用更大的批次大小：")
    config = llama2_13b_config({K: 8, B: 256})  # 增加批次大小
    k8_bs_num_params, k8_bs_memory, k8_bs_latency, k8_bs_throughput = compute_transformer_stats(config)  # @inspect k8_bs_memory, @inspect k8_bs_latency, @inspect k8_bs_throughput
    text("更差的延迟，但更好的吞吐量（而且现在适合内存了！）。")

    text("检查准确性是否下降："); link(gqa)
    image("images/gqa-accuracy.png", width=800)

    text("### Multi-head latent attention (MLA) "), link(mla)
    image("images/mla-schema.png", width=800)
    text("关键想法：将每个 key 和 value 向量从 N*H 维投影到 C 维")
    text("DeepSeek v2：将 N*H = 16384 减少到 C = 512")
    text("问题：MLA 与 RoPE 不兼容，因此需要为 RoPE 添加额外的 64 维，因此总共 512 + 64 = 576 维")
    text("延迟/吞吐量改进与之前讨论的 KV cache 减少类似")

    text("现在让我们检查准确性。")
    text("首先，MHA 比 GQA 更好（尽管更昂贵）[表 8] "); link(mla)
    image("images/mla-accuracy.png", width=800)
    text("其次，MLA 比 MHA 稍好（而且便宜得多）[表 9] "); link(mla)
    image("images/mla-accuracy2.png", width=800)

    text("### Cross-layer attention (CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("想法：在**层**之间共享 KV（就像 GQA 在头之间共享 KV 一样）")
    text("经验上改进了准确性和 KV cache 大小（延迟和吞吐量）的帕累托前沿")
    image("images/cla-results.png", width=700)

    text("### Local attention "), link(longformer), link(sparse_transformer), link(mistral_7b)
    image("images/longformer-attention.png", width=800)
    text("想法：只查看局部上下文，这对建模最相关")
    text("有效上下文随层数线性扩展")
    text("KV cache 与序列长度无关！")

    text("问题：这仍然可能损害准确性")
    text("解决方案：将局部 attention 与全局 attention 交错（混合层）")
    text("示例：character.ai 每 6 层使用 1 个全局层（除了 CLA）"), article_link("https://research.character.ai/optimizing-inference/")
    image("https://research.character.ai/content/images/2024/06/figure1-2-1.png", width=800)

    text("总结：")
    text("- 目标：在不损害准确性的情况下减少 KV cache 大小（因为推理是内存受限的）")
    text("- 低维 KV cache（GQA、MLA、共享 KV cache）")
    text("- 在某些层上使用局部 attention")


def alternatives_to_the_transformer():
    text("我们已经证明，通过调整 Transformer 的架构，我们可以改进延迟和吞吐量。")
    text("Attention + 自回归从根本上是内存受限的（Transformer 在设计时没有考虑推理）。")
    text("如果我们超越 Transformer，我们能否大幅改进？")
    text("我们将讨论两个方向：状态空间模型和扩散模型。")

    text("## 状态空间模型")
    link(title="[presentation from CS229S]", url="https://docs.google.com/presentation/d/1wrQO4uzwWr73SGj7aFxeVR9Cz0PY-mzJipn12enM39k/edit#slide=id.p")
    text("- 想法：从信号处理到在次二次时间内建模长上下文序列")
    text("- S4：基于经典状态空间模型，擅长合成长上下文任务 "), link("https://arxiv.org/abs/2111.00396")
    image("images/s4-summary.png", width=800)
    text("- 弱点：不擅长解决对语言重要的关联回忆任务（Transformer 擅长的地方）")
    image("images/based-associative-recall.png", width=400)
    text("- Mamba：允许 SSM 参数依赖于输入，在 1B 规模上匹配 Transformer "), link("https://arxiv.org/abs/2312.00752")
    text("- Jamba：交错 Transformer-Mamba 层（1:7 比例），使用 52B MoE "), link("https://arxiv.org/abs/2403.19887")
    image("images/jamba-architecture.png", width=400)
    text("- BASED：使用线性 attention + 局部 attention "), link("https://arxiv.org/abs/2402.18668")
    image("images/based-attention.png", width=400)
    text("- MiniMax-01：使用线性 attention + 完整 attention（456B 参数 MoE）"), link("https://arxiv.org/pdf/2501.08313")

    text("要点：")
    text("- 线性 + 局部 attention（仍然需要一些完整 attention）产生严肃的 SOTA 模型")
    text("- 用 O(1) 状态替换 O(T) KV cache => 推理效率更高")

    text("### 扩散模型")
    text("- 在图像生成中很流行，但在文本生成中更难工作 "), link("https://arxiv.org/abs/2205.14217")
    image("images/diffusion-lm.png", width=700)
    text("- 想法：并行生成每个 token（非自回归），多个时间步骤细化")
    text("- 从随机噪声开始（在整个序列上），迭代细化")
    text("- 来自 Inception Labs 的结果 "), article_link("https://www.inceptionlabs.ai/news")
    link(title="[demo video]", url="https://x.com/i/status/1894847919624462794")
    text("在编码基准测试中快得多：")
    image("https://framerusercontent.com/images/K2zvhtaTsz5ehDFoWx6KQHOqCyk.jpg", width=800)

    text("总体而言，通过更激进的架构变化可以在推理中获得显著收益！")


def quantization():
    text("关键想法：降低数字的精度")
    text("更少的内存意味着更高的延迟/吞吐量（因为推理是内存受限的）。")
    text("当然我们必须担心准确性...")

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    text("- fp32 (4 字节)：训练期间参数和优化器状态所需")
    text("- bf16 (2 字节)：推理的默认值")
    text("- fp8 (1 字节) [-240, 240] 用于 H100 上的 e4m3：如果你敢的话可以训练 "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8 (1 字节) [-128, 127]：不如 fp8 准确但更便宜，但仅用于推理 "), link("https://arxiv.org/pdf/2303.17951")
    text("- int4 (0.5 字节) [-8, 7]：更便宜，准确性更低 "), link("https://arxiv.org/pdf/2303.17951")

    text("Quantization-aware training (QAT)：使用量化训练，但不能扩展")
    text("Post-training quantization (PTQ)：在样本数据上运行以确定每层或张量的缩放和零点")
    link(title="[Overview of approaches]", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("### LLM.int8()")
    link("https://arxiv.org/abs/2208.07339"), article_link("https://huggingface.co/blog/hf-bitsandbytes-integration")
    text("标准量化（按绝对值的最大值缩放）：")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quant-freeze.png", width=500)
    text("问题：异常值（出现在较大的网络中）会搞砸一切")
    text("解决方案：提取异常值并在 fp16 中处理它们")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Mixed-int8.gif", width=600)
    text("效果很好（但比 fp16 慢 15-23%）：")
    image("images/llm-int8-bloom.png", width=500)

    text("### Activation-aware quantization")
    link("https://arxiv.org/abs/2306.00978")
    text("想法：根据激活选择哪些权重（0.1-1%）保持高精度")
    text("fp16 -> int3 产生 4 倍更低的内存，3.2 倍加速")
    image("images/awq-schema.png", width=800)


def model_pruning():
    text("关键想法：只需撕掉昂贵模型的部分以使其更便宜")
    text("...然后修复它。")

    text("来自 NVIDIA 的论文 "), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("算法：")
    text("1. 在小型校准数据集（1024 个样本）上识别重要的 {层、头、隐藏维度}")
    text("2. 删除不重要的层以获得更小的模型")
    text("3. 将原始模型蒸馏到修剪后的模型")

    text("结果：")
    image("images/pruning-kd.png", width=500)


def speculative_sampling():
    text("回顾推理的两个阶段：")
    text("- Prefill：给定一个序列，并行编码 token（计算受限）[注意：也给你概率]")
    text("- Generation：一次生成一个 token（内存受限）")
    text("换句话说，检查比生成更快。")

    text("Speculative sampling "); link("https://arxiv.org/abs/2211.17192"); link("https://arxiv.org/abs/2302.01318")
    text("- 使用更便宜的 **draft model** p 猜测几个 token（例如 4 个）")
    text("- 使用目标模型 q 评估（并行处理 token），如果看起来不错就接受")
    link(title="[Speculative sampling video]", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("这是修改后的拒绝采样，提案为 p，目标为 q")
    text("修改：始终生成至少一个候选（拒绝采样会一直循环）")
    text("关键属性：保证是目标模型的**精确样本**！")

    text("通过示例证明：假设两个词汇元素 {A, B}")
    text("- 目标模型概率：[q(A), q(B)]")
    text("- Draft 模型概率：[p(A), p(B)]")
    text("- 假设 p(A) > q(A) [draft 模型过度采样 A]。")
    text("- 因此 p(B) < q(B) [draft 模型欠采样 B]。")
    text("- 残差概率 max(q-p, 0): [0, 1]")
    text("计算推测采样 token 的概率：")
    text("- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)")
    text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    text("实践中：")
    text("- 目标模型有 70B 参数，draft 模型有 8B 参数")
    text("- 目标模型有 8B 参数，draft 模型有 1B 参数")
    text("- 尝试使 draft 模型尽可能接近目标（蒸馏）")

    text("改进 draft 模型的扩展：")
    text("- Medusa：draft 模型并行生成多个 token "), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE：draft 模型从目标模型获取高级特征 "), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    text("总结：")
    text("- 从目标模型精确采样（感谢数学！）")
    text("- 利用检查和生成之间的不对称性")
    text("- draft 模型有很大的创新空间（涉及训练）")


def continuous_batching():
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="[talk]", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("问题：")
    text("- 训练：获得密集的 token 块（批次大小 x 序列长度）")
    text("- 推理：请求在不同时间到达和完成，所以你有一个不规则数组")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    text("解决方案：迭代级调度")
    text("- 逐步解码")
    text("- 在新请求到达时将其添加到批次中（因此不必等到生成完成）")

    text("问题：")
    text("- 批处理仅在所有序列具有相同维度时才有效（对吗？）")
    text("- 但每个请求可能有不同的长度")

    text("解决方案：选择性批处理")
    text("- 训练：当所有序列长度相同时，对 B x S x H 张量进行操作")
    text("- 但我们可能有不同的长度：[3, H], [9, H], [5, H] 等。")
    text("- Attention 计算：分别处理每个序列")
    text("- 非 attention 计算：将所有序列连接在一起到 [3 + 9 + 5, H]")


def paged_attention():
    text("引入 vLLM 和 PagedAttention 的论文 "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("以前的现状：")
    text("- 请求进来")
    text("- 为提示和响应分配 KV cache 部分（最多到最大长度）")
    image("images/paged-attention-fragmentation.png", width=800)
    text("问题：碎片化（你的硬盘会发生什么）")
    text("- 但这是浪费的，因为我们可能生成更少的 token（内部碎片）！")
    text("- 部分之间可能有额外的未使用空间（外部碎片）！")

    text("解决方案：PagedAttention（记住操作系统）")
    text("- 将序列的 KV cache 分成非连续的**块**")
    image("images/paged-attention-blocks.png", width=400)

    text("两个请求共享 KV cache：")
    image("images/paged-attention-logical.png", width=800)

    text("一般来说，跨序列共享 KV cache 的多种类型：")
    image("images/paged-attention-sharing.png", width=600)
    text("- 共享系统提示")
    text("- 每个提示采样多个响应（例如，用于程序合成）")

    text("解决方案：共享前缀，在块级别写时复制")
    image("images/paged-attention-parallel.png", width=600)

    text("其他 vLLM 优化：")
    text("- 融合块读取和 attention 的 kernel（减少 kernel 启动开销）")
    text("- 使用最新的 kernel（FlashAttention、FlashDecoding）")
    text("- 使用 CUDA 图避免 kernel 启动开销")

    text("总结：使用操作系统的想法（分页）来利用内存处理动态工作负载")


if __name__ == "__main__":
    main()
