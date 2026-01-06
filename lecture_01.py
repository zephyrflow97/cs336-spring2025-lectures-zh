import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random

from execute_util import link, image, text
from lecture_util import article_link, x_link, youtube_link
from references import gpt_3, gpt4, shannon1950, bengio2003, susketver2014, \
    bahdanau2015_attention, transformer_2017, gpt2, t5, kaplan_scaling_laws_2020, \
    the_pile, gpt_j, opt_175b, bloom, palm, chinchilla, llama, mistral_7b, \
    instruct_gpt, dpo, adamw2017, lima, deepseek_v3, adam2014, grpo, ppo2017, muon, \
    large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, olmo_7b, moe_2017, \
    megatron_lm_2019, shazeer_2020, elmo, bert, qwen_2_5, deepseek_r1, moe_2017, \
    rms_norm_2019, rope_2021, soap, gqa, mla, deepseek_67b, deepseek_v2, brants2007, \
    layernorm_2016, pre_post_norm_2020, llama2, llama3, olmo2, \
    megabyte, byt5, blt, tfree, sennrich_2016, zero_2019, gpipe_2018
from data import get_common_crawl_urls, read_common_crawl, write_documents, markdownify_documents
from model_util import query_gpt4o

import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_landscape()

    what_is_this_program()

    course_logistics()
    course_components()

    tokenization()

    text("下次课程：PyTorch 构建模块，资源核算")


def welcome():
    text("## CS336: 从零开始构建语言模型 (2025年春季)"),

    image("images/course-staff.png", width=600)

    text("这是 CS336 的第二次开课。")
    text("斯坦福版本已经增长了 50%。")
    text("讲座将发布在 YouTube 上，并向全世界开放。")


def why_this_course_exists():
    text("## 为什么我们要开设这门课程？")

    text("让我们问问 GPT-4 "), link(gpt4)
    response = query_gpt4o(prompt="Why teach a course on building language models from scratch? Answer in one sentence.")  # @inspect response
    
    text("问题：研究人员正在与底层技术**脱节**。")
    text("8 年前，研究人员会实现并训练自己的模型。")
    text("6 年前，研究人员会下载一个模型（例如 BERT）并对其进行微调。")
    text("今天，研究人员只是提示一个专有模型（例如 GPT-4/Claude/Gemini）。")

    text("提升抽象层次可以提高生产力，但是")
    text("- 这些抽象是有漏洞的（与编程语言或操作系统相比）。")
    text("- 仍然有需要深入底层的基础研究工作要做。")

    text("**全面理解**这项技术对于**基础研究**是必要的。")

    text("本课程：**通过构建来理解**")
    text("但有一个小问题...")

    text("## 语言模型的工业化")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Industrialisation.jpg/440px-Industrialisation.jpg", width=400)

    text("据称 GPT-4 有 1.8T 参数。"), article_link("https://www.hpcwire.com/2024/03/19/the-generative-ai-future-is-now-nvidias-huang-says")
    text("据称 GPT-4 的训练成本为 1 亿美元。"), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("xAI 构建了拥有 200,000 个 H100 的集群来训练 Grok。"), article_link("https://www.tomshardware.com/pc-components/gpus/elon-musk-is-doubling-the-worlds-largest-ai-gpu-cluster-expanding-colossus-gpu-cluster-to-200-000-soon-has-floated-300-000-in-the-past")
    text("Stargate（OpenAI、NVIDIA、Oracle）在 4 年内投资 5000 亿美元。"), article_link("https://openai.com/index/announcing-the-stargate-project/")

    text("此外，前沿模型的构建方式没有公开细节。")
    text("来自 GPT-4 技术报告 "), link(gpt4), text("：")
    image("images/gpt4-no-details.png", width=600)

    text("## 规模不同，性质不同")
    text("前沿模型对我们来说遥不可及。")
    text("但构建小型语言模型（本课程中 <1B 参数）可能无法代表大型语言模型。")

    text("示例 1：attention 与 MLP 中花费的 FLOPs 比例随规模变化。"), x_link("https://x.com/stephenroller/status/1579993017234382849")
    image("images/roller-flops.png", width=400)
    text("示例 2：随规模出现的涌现行为 "), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)

    text("## 在这门课中我们能学到什么可以迁移到前沿模型？")
    text("有三种类型的知识：")
    text("- **机制（Mechanics）**：事物如何工作（什么是 Transformer，模型并行如何利用 GPU）")
    text("- **思维方式（Mindset）**：充分利用硬件，认真对待规模（scaling laws）")
    text("- **直觉（Intuitions）**：哪些数据和建模决策能产生良好的准确性")

    text("我们可以教授机制和思维方式（这些可以迁移）。")
    text("我们只能部分教授直觉（不一定能跨规模迁移）。")

    text("## 直觉？🤷")
    text("有些设计决策（目前）无法证明合理性，只是来自实验。")
    text("示例：Noam Shazeer 引入 SwiGLU 的论文 "), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)

    text("## 痛苦的教训（The bitter lesson）")
    text("错误的理解：规模就是一切，算法不重要。")
    text("正确的理解：能够扩展的算法才是重要的。")
    text("### 准确性 = 效率 × 资源")
    text("事实上，在更大规模下效率更加重要（不能浪费）。")
    link("https://arxiv.org/abs/2005.04305"), text(" 显示在 2012 到 2019 年间，ImageNet 上的算法效率提高了 44 倍")

    text("框架：在给定的计算和数据预算下，能构建的最佳模型是什么？")
    text("换句话说，**最大化效率**！")


def current_landscape():
    text("## 神经网络之前（2010年代之前）")
    text("- 用于测量英语熵的语言模型 "), link(shannon1950)
    text("- 大量关于 n-gram 语言模型的工作（用于机器翻译、语音识别）"), link(brants2007)

    text("## 神经网络组件（2010年代）")
    text("- 第一个神经语言模型 "), link(bengio2003)
    text("- Sequence-to-sequence 建模（用于机器翻译）"), link(susketver2014)
    text("- Adam 优化器 "), link(adam2014)
    text("- Attention 机制（用于机器翻译）"), link(bahdanau2015_attention)
    text("- Transformer 架构（用于机器翻译）"), link(transformer_2017)
    text("- Mixture of experts "), link(moe_2017)
    text("- 模型并行 "), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)

    text("## 早期基础模型（2010年代末）")
    text("- ELMo：使用 LSTM 预训练，微调有助于任务 "), link(elmo)
    text("- BERT：使用 Transformer 预训练，微调有助于任务 "), link(bert)
    text("- Google 的 T5 (11B)：将所有任务转换为 text-to-text "), link(t5)

    text("## 拥抱规模，更加封闭")
    text("- OpenAI 的 GPT-2 (1.5B)：流畅的文本，首次出现 zero-shot 迹象，分阶段发布 "), link(gpt2)
    text("- Scaling laws：为扩展提供希望/可预测性 "), link(kaplan_scaling_laws_2020)
    text("- OpenAI 的 GPT-3 (175B)：in-context learning，封闭 "), link(gpt_3)
    text("- Google 的 PaLM (540B)：大规模，训练不足 "), link(palm)
    text("- DeepMind 的 Chinchilla (70B)：计算最优 scaling laws "), link(chinchilla)

    text("## 开放模型")
    text("- EleutherAI 的开放数据集（The Pile）和模型（GPT-J）"), link(the_pile), link(gpt_j)
    text("- Meta 的 OPT (175B)：GPT-3 复现，许多硬件问题 "), link(opt_175b)
    text("- Hugging Face / BigScience 的 BLOOM：专注于数据来源 "), link(bloom)
    text("- Meta 的 Llama 模型 "), link(llama), link(llama2), link(llama3)
    text("- 阿里巴巴的 Qwen 模型 "), link(qwen_2_5)
    text("- DeepSeek 的模型 "), link(deepseek_67b), link(deepseek_v2), link(deepseek_v3)
    text("- AI2 的 OLMo 2 "), link(olmo_7b), link(olmo2),

    text("## 开放程度")
    text("- 封闭模型（例如 GPT-4o）：仅 API 访问 "), link(gpt4)
    text("- 开放权重模型（例如 DeepSeek）：权重可用，论文包含架构细节，一些训练细节，无数据细节 "), link(deepseek_v3)
    text("- 开源模型（例如 OLMo）：权重和数据可用，论文包含大部分细节（但不一定包括理由、失败的实验）"), link(olmo_7b)

    text("## 当今的前沿模型")
    text("- OpenAI 的 o3 "), link("https://openai.com/index/openai-o3-mini/")
    text("- Anthropic 的 Claude Sonnet 3.7 "), link("https://www.anthropic.com/news/claude-3-7-sonnet")
    text("- xAI 的 Grok 3 "), link("https://x.ai/news/grok-3")
    text("- Google 的 Gemini 2.5 "), link("https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/")
    text("- Meta 的 Llama 3.3 "), link("https://ai.meta.com/blog/meta-llama-3/")
    text("- DeepSeek 的 r1 "), link(deepseek_r1)
    text("- 阿里巴巴的 Qwen 2.5 Max "), link("https://qwenlm.github.io/blog/qwen2.5-max/")
    text("- 腾讯的 Hunyuan-T1 "), link("https://tencent.github.io/llm.hunyuan.T1/README_EN.html")


def what_is_this_program():
    text("这是一个*可执行讲座*，一个通过执行来传递讲座内容的程序。")
    text("可执行讲座使以下操作成为可能：")
    text("- 查看和运行代码（因为一切都是代码！），")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- 查看讲座的层次结构，以及")
    text("- 跳转到定义和概念："), link(supervised_finetuning)


def course_logistics():
    text("所有信息都在线上："), link("https://stanford-cs336.github.io/spring2025/")

    text("这是一门 5 学分的课程。")
    text("来自 2024 年春季课程评估的评论：*整个作业的工作量大约相当于 CS 224n 的全部 5 个作业加上最终项目。而这只是第一个作业。*")

    text("## 为什么你应该选这门课")
    text("- 你有强烈的需求去理解事物的工作原理。")
    text("- 你想锻炼研究工程能力。")

    text("## 为什么你不应该选这门课")
    text("- 你实际上想在本季度完成研究工作。<br>（和你的导师谈谈。）")
    text("- 你对学习 AI 中最热门的新技术感兴趣（例如多模态、RAG 等）。<br>（你应该选一门研讨课。）")
    text("- 你想在自己的应用领域获得良好结果。<br>（你应该只需提示或微调现有模型。）")

    text("## 如何在家跟随学习")
    text("- 所有讲座材料和作业都将在线发布，所以可以自由跟随学习。")
    text("- 讲座通过 [CGOE，正式名称 SCPD](https://cgoe.stanford.edu/) 录制，并在 YouTube 上提供（会有一些延迟）。")
    text("- 我们计划明年再次开设这门课。")

    text("## 作业")
    text("- 5 个作业（基础、系统、scaling laws、数据、对齐）。")
    text("- 没有脚手架代码，但我们提供单元测试和适配器接口来帮助你检查正确性。")
    text("- 在本地实现以测试正确性，然后在集群上运行以进行基准测试（准确性和速度）。")
    text("- 某些作业有排行榜（在给定训练预算下最小化困惑度）。")
    text("- AI 工具（例如 CoPilot、Cursor）可能会影响学习，所以使用时需自担风险。")

    text("## 集群")
    text("- 感谢 Together AI 提供计算集群。🙏")
    text("- 请阅读[指南](https://docs.google.com/document/d/1BSSig7zInyjDKcbNGftVxubiHlwJ-ZqahQewIzBmBOo/edit)了解如何使用集群。")
    text("- 尽早开始作业，因为临近截止日期时集群会被占满！")


def course_components():
    text("## 一切都关乎效率")
    text("资源：数据 + 硬件（计算、内存、通信带宽）")
    text("在给定的固定资源集下，如何训练最佳模型？")
    text("示例：给定一个 Common Crawl 转储和 32 个 H100，持续 2 周，你应该怎么做？")

    text("设计决策：")
    image("images/design-decisions.png", width=800)

    text("## 课程概览")
    basics()
    systems()
    scaling_laws()
    data()
    alignment()

    text("## 效率驱动设计决策")

    text("今天，我们受计算约束，因此设计决策将反映如何充分利用给定硬件。")
    text("- 数据处理：避免在糟糕/无关的数据上浪费宝贵的计算资源")
    text("- Tokenization：使用原始字节很优雅，但在当今的模型架构下计算效率低下。")
    text("- 模型架构：许多变化是为了减少内存或 FLOPs（例如共享 KV 缓存、滑动窗口 attention）")
    text("- 训练：我们可以只用一个 epoch！")
    text("- Scaling laws：在较小模型上使用更少计算来进行超参数调优")
    text("- 对齐：如果将模型更多地调整到所需用例，则需要更小的基础模型")

    text("明天，我们将受到数据约束...")


class Tokenizer(ABC):
    """Tokenizer 的抽象接口。"""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    text("目标：让完整流水线的基本版本运行起来")
    text("组件：tokenization、模型架构、训练")

    text("## Tokenization")
    text("Tokenizer 在字符串和整数序列（token）之间进行转换")
    image("images/tokenized-example.png", width=600) 
    text("直觉：将字符串分解为常见片段")

    text("本课程：Byte-Pair Encoding (BPE) tokenizer "), link(sennrich_2016)

    text("无 tokenizer 方法："), link(byt5), link(megabyte), link(blt), link(tfree)
    text("直接使用字节，很有前景，但尚未扩展到前沿水平。")
    
    text("## 架构")
    text("起点：原始 Transformer "), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)

    text("变体：")
    text("- 激活函数：ReLU、SwiGLU "), link(shazeer_2020)
    text("- 位置编码：sinusoidal、RoPE "), link(rope_2021)
    text("- 归一化：LayerNorm、RMSNorm "), link(layernorm_2016), link(rms_norm_2019)
    text("- 归一化的位置：pre-norm 与 post-norm "), link(pre_post_norm_2020)
    text("- MLP：dense、mixture of experts "), link(moe_2017)
    text("- Attention：full、sliding window、linear "), link(mistral_7b), link("https://arxiv.org/abs/2006.16236")
    text("- 低维 attention：group-query attention (GQA)、multi-head latent attention (MLA) "), link(gqa), link(mla)
    text("- 状态空间模型：Hyena "), link("https://arxiv.org/abs/2302.10866")

    text("## 训练")
    text("- 优化器（例如 AdamW、Muon、SOAP）"), link(adam2014), link(adamw2017), link(muon), link(soap)
    text("- 学习率调度（例如 cosine、WSD）"), link(cosine_learning_rate_2017), link(wsd_2024)
    text("- Batch size（例如临界 batch size）"), link(large_batch_training_2018)
    text("- 正则化（例如 dropout、weight decay）")
    text("- 超参数（head 数量、隐藏维度）：网格搜索")

    text("## 作业 1")
    link(title="[GitHub]", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="[PDF]", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf")
    text("- 实现 BPE tokenizer")
    text("- 实现 Transformer、交叉熵损失、AdamW 优化器、训练循环")
    text("- 在 TinyStories 和 OpenWebText 上训练")
    text("- 排行榜：在 H100 上 90 分钟内最小化 OpenWebText 困惑度 "), link(title="[去年的排行榜]", url="https://github.com/stanford-cs336/spring2024-assignment1-basics-leaderboard")


def systems():
    text("目标：充分利用硬件")
    text("组件：kernel、并行、推理")

    text("## Kernel")
    text("GPU (A100) 的样子：")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=800)
    text("类比：仓库 : DRAM :: 工厂 : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png", width=400)
    text("技巧：通过最小化数据移动来组织计算，以最大化 GPU 利用率")
    text("使用 CUDA/**Triton**/CUTLASS/ThunderKittens 编写 kernel")

    text("## 并行")
    text("如果我们有多个 GPU（8 个 A100）呢？")
    image("https://www.fibermall.com/blog/wp-content/uploads/2024/09/the-hardware-topology-of-a-typical-8xA100-GPU-host.png", width=500)
    text("GPU 之间的数据移动更慢，但同样的'最小化数据移动'原则仍然适用")
    text("使用集合操作（例如 gather、reduce、all-reduce）")
    text("跨 GPU 分片（参数、激活、梯度、优化器状态）")
    text("如何拆分计算：{data, tensor, pipeline, sequence} 并行")
    
    text("## 推理")
    text("目标：给定提示生成 token（实际使用模型所需！）")
    text("推理也需要用于强化学习、测试时计算、评估")
    text("全球范围内，推理计算（每次使用）超过训练计算（一次性成本）")
    text("两个阶段：prefill 和 decode")
    image("images/prefill-decode.png", width=500)
    text("Prefill（类似于训练）：token 已给定，可以一次处理所有（计算受限）")
    text("Decode：需要一次生成一个 token（内存受限）")
    text("加速解码的方法：")
    text("- 使用更便宜的模型（通过模型剪枝、量化、蒸馏）")
    text("- Speculative decoding：使用更便宜的\"草稿\"模型生成多个 token，然后使用完整模型并行评分（精确解码！）")
    text("- 系统优化：KV 缓存、批处理")

    text("## 作业 2")
    link(title="[2024年的 GitHub]", url="https://github.com/stanford-cs336/spring2024-assignment2-systems"), link(title="[2024年的 PDF]", url="https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf")
    text("- 在 Triton 中实现融合的 RMSNorm kernel")
    text("- 实现分布式数据并行训练")
    text("- 实现优化器状态分片")
    text("- 对实现进行基准测试和性能分析")


def scaling_laws():
    text("目标：在小规模上做实验，预测大规模的超参数/损失")
    text("问题：给定 FLOPs 预算（$C$），使用更大的模型（$N$）还是在更多 token 上训练（$D$）？")
    text("计算最优 scaling laws："), link(kaplan_scaling_laws_2020), link(chinchilla)
    image("images/chinchilla-isoflop.png", width=800)
    text("简而言之：$D^* = 20 N^*$（例如，1.4B 参数模型应该在 28B token 上训练）")
    text("但这没有考虑推理成本！")

    text("## 作业 3")
    link(title="[2024年的 GitHub]", url="https://github.com/stanford-cs336/spring2024-assignment3-scaling"), link(title="[2024年的 PDF]", url="https://github.com/stanford-cs336/spring2024-assignment3-scaling/blob/master/cs336_spring2024_assignment3_scaling.pdf")
    text("- 我们基于之前的运行定义一个训练 API（超参数 -> 损失）")
    text("- 提交\"训练任务\"（在 FLOPs 预算下）并收集数据点")
    text("- 将 scaling law 拟合到数据点")
    text("- 提交扩展超参数的预测")
    text("- 排行榜：在给定 FLOPs 预算下最小化损失")


def data():
    text("问题：我们希望模型具有什么能力？")
    text("多语言？代码？数学？")
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)

    text("## 评估")
    text("- 困惑度（Perplexity）：语言模型的教科书式评估")
    text("- 标准化测试（例如 MMLU、HellaSwag、GSM8K）")
    text("- 指令遵循（例如 AlpacaEval、IFEval、WildBench）")
    text("- 扩展测试时计算：chain-of-thought、集成")
    text("- LM-as-a-judge：评估生成任务")
    text("- 完整系统：RAG、agent")

    text("## 数据策划")
    text("- 数据不会从天而降。")
    look_at_web_data()
    text("- 来源：从互联网爬取的网页、书籍、arXiv 论文、GitHub 代码等。")
    text("- 诉诸合理使用来训练版权数据？"), link("https://arxiv.org/pdf/2303.15715.pdf")
    text("- 可能需要授权数据（例如 Google 与 Reddit 数据）"), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- 格式：HTML、PDF、目录（不是文本！）")

    text("## 数据处理")
    text("- 转换：将 HTML/PDF 转换为文本（保留内容、一些结构、重写）")
    text("- 过滤：保留高质量数据，删除有害内容（通过分类器）")
    text("- 去重：节省计算，避免记忆；使用 Bloom filter 或 MinHash")

    text("## 作业 4")
    link(title="[2024年的 GitHub]", url="https://github.com/stanford-cs336/spring2024-assignment4-data"), link(title="[2024年的 PDF]", url="https://github.com/stanford-cs336/spring2024-assignment4-data/blob/master/cs336_spring2024_assignment4_data.pdf")
    text("- 将 Common Crawl HTML 转换为文本")
    text("- 训练分类器以过滤质量和有害内容")
    text("- 使用 MinHash 去重")
    text("- 排行榜：在给定 token 预算下最小化困惑度")


def look_at_web_data():
    urls = get_common_crawl_urls()[:3]  # @inspect urls
    documents = list(read_common_crawl(urls[1], limit=300))
    random.seed(40)
    random.shuffle(documents)
    documents = markdownify_documents(documents[:10])
    write_documents(documents, "var/sample-documents.txt")
    link(title="[示例文档]", url="var/sample-documents.txt")
    text("外面是一片荒地！需要真正处理数据。")


def alignment():
    text("到目前为止，**基础模型**是原始潜力，非常擅长完成下一个 token。")
    text("对齐使模型真正有用。")

    text("对齐的目标：")
    text("- 让语言模型遵循指令")
    text("- 调整风格（格式、长度、语气等）")
    text("- 纳入安全性（例如拒绝回答有害问题）")

    text("两个阶段：")
    supervised_finetuning()
    learning_from_feedback()

    text("## 作业 5")
    link(title="[2024年的 GitHub]", url="https://github.com/stanford-cs336/spring2024-assignment5-alignment"), link(title="[2024年的 PDF]", url="https://github.com/stanford-cs336/spring2024-assignment5-alignment/blob/master/cs336_spring2024_assignment5_alignment.pdf")
    text("- 实现监督微调")
    text("- 实现 Direct Preference Optimization (DPO)")
    text("- 实现 Group Relative Preference Optimization (GRPO)")


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class ChatExample:
    turns: list[Turn]


@dataclass(frozen=True)
class PreferenceExample:
    history: list[Turn]
    response_a: str
    response_b: str
    chosen: str


def supervised_finetuning():
    text("## 监督微调（Supervised finetuning, SFT）")

    text("指令数据：（提示，响应）对")
    sft_data: list[ChatExample] = [
        ChatExample(
            turns=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is 1 + 1?"),
                Turn(role="assistant", content="The answer is 2."),
            ],
        ),
    ]
    text("数据通常涉及人工标注。")
    text("直觉：基础模型已经具备技能，只需要少量示例来展现它们。"), link(lima)
    text("监督学习：微调模型以最大化 p(response | prompt)。")


def learning_from_feedback():
    text("现在我们有了一个初步的指令遵循模型。")
    text("让我们在不进行昂贵标注的情况下改进它。")
    
    text("## 偏好数据")
    text("数据：使用模型对给定提示生成多个响应（例如 [A, B]）。")
    text("用户提供偏好（例如 A < B 或 A > B）。")
    preference_data: list[PreferenceExample] = [
        PreferenceExample(
            history=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is the best way to train a language model?"),
            ],
            response_a="You should use a large dataset and train for a long time.",
            response_b="You should use a small dataset and train for a short time.",
            chosen="a",
        )
    ]

    text("## 验证器")
    text("- 形式化验证器（例如用于代码、数学）")
    text("- 学习的验证器：针对 LM-as-a-judge 进行训练")

    text("## 算法")
    text("- 来自强化学习的 Proximal Policy Optimization (PPO) "), link(ppo2017), link(instruct_gpt)
    text("- Direct Policy Optimization (DPO)：用于偏好数据，更简单 "), link(dpo)
    text("- Group Relative Preference Optimization (GRPO)：移除 value function "), link(grpo)


############################################################
# Tokenization

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tokenization():
    text("本单元受 Andrej Karpathy 关于 tokenization 的视频启发；去看看吧！"), youtube_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("## 总结")
    text("- Tokenizer：字符串 <-> token（索引）")
    text("- 基于字符、基于字节、基于单词的 tokenization 高度次优")
    text("- BPE 是一种有效的启发式方法，查看语料库统计信息")
    text("- Tokenization 是一个必要的恶，也许有一天我们只需从字节开始...")

@dataclass(frozen=True)
class BPETokenizerParams:
    """指定 BPETokenizer 所需的全部内容。"""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



class CharacterTokenizer(Tokenizer):
    """将字符串表示为 Unicode 码点序列。"""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """将字符串表示为字节序列。"""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """返回 `indices`，但将所有 `pair` 实例替换为 `new_index`。"""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    """给定一组合并和词汇表的 BPE tokenizer。"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # 注意：这是一个非常慢的实现
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """给定已被 tokenize 为 `indices` 的 `string`，计算压缩比。"""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")


def intro_to_tokenization():
    text("原始文本通常表示为 Unicode 字符串。")
    string = "Hello, 🌍! 你好!"

    text("语言模型在 token 序列上放置概率分布（通常由整数索引表示）。")
    indices = [15496, 11, 995, 0]

    text("所以我们需要一个将字符串*编码*为 token 的过程。")
    text("我们还需要一个将 token *解码*回字符串的过程。")
    text("一个 "), link(Tokenizer), text(" 是实现 encode 和 decode 方法的类。")
    text("**词汇表大小**是可能的 token（整数）数量。")


def tokenization_examples():
    text("要了解 tokenizer 的工作原理，请使用这个 "), link(title="交互式网站", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## 观察")
    text("- 一个单词及其前面的空格是同一个 token 的一部分（例如 \" world\"）。")
    text("- 开头和中间的单词表示方式不同（例如 \"hello hello\"）。")
    text("- 数字被 tokenize 为每几位数字。")

    text("这是来自 OpenAI 的 GPT-2 tokenizer（tiktoken）的实际应用。")
    tokenizer = get_gpt2_tokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("检查 encode() 和 decode() 是否往返：")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def character_tokenizer():
    text("## 基于字符的 tokenization")

    text("Unicode 字符串是 Unicode 字符的序列。")
    text("每个字符可以通过 `ord` 转换为码点（整数）。")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("可以通过 `chr` 转换回来。")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("现在让我们构建一个 `Tokenizer` 并确保它往返：")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("大约有 150K 个 Unicode 字符。"), link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # 这是一个下界 @inspect vocabulary_size
    text("问题 1：这是一个非常大的词汇表。")
    text("问题 2：许多字符相当罕见（例如 🌍），这是词汇表的低效使用。")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def byte_tokenizer():
    text("## 基于字节的 tokenization")

    text("Unicode 字符串可以表示为字节序列，可以用 0 到 255 之间的整数表示。")
    text("最常见的 Unicode 编码是 "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("一些 Unicode 字符由一个字节表示：")
    assert bytes("a", encoding="utf-8") == b"a"
    text("其他字符需要多个字节：")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("现在让我们构建一个 `Tokenizer` 并确保它往返：")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("词汇表又好又小：一个字节可以表示 256 个值。")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("压缩率如何？")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1
    text("压缩比很糟糕，这意味着序列会太长。")
    text("考虑到 Transformer 的上下文长度是有限的（因为 attention 是二次的），这看起来不太好...")


def word_tokenizer():
    text("## 基于单词的 tokenization")

    text("另一种方法（更接近 NLP 中经典做法）是将字符串拆分为单词。")
    string = "I'll say supercalifragilisticexpialidocious!"

    segments = regex.findall(r"\w+|.", string)  # @inspect segments
    text("这个正则表达式将所有字母数字字符保持在一起（单词）。")

    text("这是一个更高级的版本：")
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments

    text("要将其转换为 `Tokenizer`，我们需要将这些片段映射为整数。")
    text("然后，我们可以构建从每个片段到整数的映射。")

    text("但存在问题：")
    text("- 单词数量巨大（就像 Unicode 字符一样）。")
    text("- 许多单词很罕见，模型不会学到太多关于它们的东西。")
    text("- 这显然不能提供固定的词汇表大小。")

    text("训练期间未见过的新单词会得到一个特殊的 UNK token，这很丑陋，并且会搞乱困惑度计算。")

    vocabulary_size = "训练数据中不同片段的数量"
    compression_ratio = get_compression_ratio(string, segments)  # @inspect compression_ratio


def bpe_tokenizer():
    text("## Byte Pair Encoding (BPE)")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Byte_pair_encoding")
    text("BPE 算法由 Philip Gage 于 1994 年引入用于数据压缩。"), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("它被改编用于神经机器翻译的 NLP。"), link(sennrich_2016)
    text("（之前，论文一直在使用基于单词的 tokenization。）")
    text("BPE 随后被 GPT-2 使用。"), link(gpt2)

    text("基本思想：在原始文本上*训练* tokenizer 以自动确定词汇表。")
    text("直觉：常见的字符序列由单个 token 表示，罕见的序列由许多 token 表示。")

    text("GPT-2 论文使用基于单词的 tokenization 将文本分解为初始片段，并在每个片段上运行原始 BPE 算法。")
    text("草图：从每个字节作为 token 开始，并连续合并最常见的相邻 token 对。")

    text("## 训练 tokenizer")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    text("## 使用 tokenizer")
    text("现在，给定一个新文本，我们可以对其进行编码。")
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("在作业 1 中，你将通过以下方式超越这一点：")
    text("- encode() 当前循环遍历所有合并。只循环重要的合并。")
    text("- 检测并保留特殊 token（例如 <|endoftext|>）。")
    text("- 使用预 tokenization（例如 GPT-2 tokenizer regex）。")
    text("- 尝试使实现尽可能快。")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("从 `string` 的字节列表开始。")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        text("计算每对 token 的出现次数")
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # 对于每个相邻对
            counts[(index1, index2)] += 1  # @inspect counts

        text("找到最常见的对。")
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair

        text("合并该对。")
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    main()
