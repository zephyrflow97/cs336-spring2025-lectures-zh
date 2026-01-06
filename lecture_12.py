from execute_util import text, link, image
from lecture_util import x_link, blog_link
from references import deepseek_r1, llama4, olmo2_32b, mmlu

def main():
    text("**评估**：给定一个**固定模型**，它有多\"**好**\"？")

    what_you_see()
    how_to_think_about_evaluation()

    perplexity()

    knowledge_benchmarks()
    instruction_following_benchmarks()
    agent_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    what_are_we_evaluating()

    text("要点")
    text("- 没有唯一正确的评估；根据你想要测量的内容选择评估。")
    text("- 始终查看单个实例和预测。")
    text("- 有许多方面需要考虑：能力、安全性、成本、真实性。")
    text("- 明确说明游戏规则（方法 vs 模型/系统）。")


def what_you_see():
    text("## 基准测试分数")
    image("images/deepseek-r1-benchmarks.png", width=800), link(deepseek_r1)
    image("images/llama4-benchmarks.png", width=800), link(llama4)
    image("https://www.datocms-assets.com/64837/1741887109-instruct-1.png", width=800), link(olmo2_32b)

    text("最近的语言模型在相似但不完全相同的 benchmark 上进行评估（MMLU、MATH 等）。")
    text("这些 benchmark 是什么？")
    text("这些数字意味着什么？")

    image("images/helm-capabilities-leaderboard.png", width=1000)
    link(title="[HELM capabilities]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard")

    text("密切关注成本！")
    image("images/artificial-analysis.png", width=800), link(title="[Artificial Analysis]", url="https://artificialanalysis.ai/")

    text("也许一个模型好不好取决于人们是否选择使用它（并为之付费）...")
    image("images/openrouter.png", width=600), link(title="[OpenRouter]", url="https://openrouter.ai/rankings")

    image("images/chatbot-arena-leaderboard.png", width=800)
    link(title="[Chatbot Arena]", url="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard")

    text("## 感觉")
    x_link("https://x.com/demishassabis/status/1919779362980692364")
    image("images/demis-gemini-2.5.png", width=500)

    text("一场危机...")
    image("images/karpathy-crisis.png", width=600)


def how_to_think_about_evaluation():
    text("你可能认为评估是一个机械的过程（拿一个现有模型，向它扔一些 prompts，平均一些数字）...")
    text("实际上，评估是一个深刻而丰富的话题...")
    text("...它决定了语言模型的未来。")

    text("评估的意义是什么？")
    text("没有唯一正确的评估；这取决于你想要回答什么问题。")
    text("1. 用户或公司想要为他们的用例（例如，客户服务聊天机器人）做出购买决策（模型 A 还是模型 B）。")
    text("2. 研究人员想要测量模型的原始能力（例如，智能）。")
    text("3. 我们想要了解模型的好处和危害（出于商业和政策原因）。")
    text("4. 模型开发者想要获得反馈以改进模型。")
    text("在每种情况下，都有一个抽象的**目标**需要转化为具体的评估。")

    text("框架")
    text("1. **输入**是什么？")
    text("2. 如何**调用**语言模型？")
    text("3. 如何评估**输出**？")
    text("4. 如何**解释**结果？")

    text("输入是什么？")
    text("1. **覆盖**了哪些用例？")
    text("2. 我们是否有长尾中**困难**输入的代表？")
    text("3. 输入是否**适应**模型（例如，多轮对话）？")

    text("如何调用语言模型？")
    text("1. 如何 prompt 语言模型？")
    text("2. 语言模型是否使用 chain-of-thought、工具、RAG 等？")
    text("3. 我们是在评估语言模型还是 agent 系统（模型开发者想要前者，用户想要后者）？")

    text("如何评估输出？")
    text("1. 用于评估的参考输出是否无错误？")
    text("2. 使用什么指标（例如，pass@k）？")
    text("3. 如何考虑成本（例如，推理 + 训练）？")
    text("4. 如何考虑不对称错误（例如，医疗环境中的幻觉）？")
    text("5. 如何处理开放式生成（没有 ground truth）？")

    text("如何解释指标？")
    text("1. 如何解释一个数字（例如，91%）- 它是否准备好部署？")
    text("2. 面对训练-测试重叠，我们如何评估泛化能力？")
    text("3. 我们是在评估最终模型还是方法？")

    text("总结：在进行评估时有很多问题需要思考")

def perplexity():
    text("回顾：语言模型是 token 序列上的概率分布 **p(x)**。")
    text("Perplexity (1/p(D))^(1/|D|) 测量 p 是否为某个数据集 D 分配高概率。")

    text("在预训练中，你在训练集上最小化 perplexity。")
    text("显而易见的是在测试集上测量 perplexity。")

    text("标准数据集：Penn Treebank (WSJ)、WikiText-103 (Wikipedia)、One Billion Word Benchmark（来自机器翻译 WMT11 - EuroParl、UN、新闻）")
    text("论文在数据集上训练（训练集）并在同一数据集上评估（测试集）")
    text("纯 CNNs+LSTMs 在 One Billion Word Benchmark 上（perplexity 51.3 -> 30.0）"), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2 在 WebText（40GB 文本，来自 Reddit 链接的网站）上训练，在标准数据集上 zero-shot")
    text("这是分布外评估（但想法是训练覆盖了很多内容）")
    image("images/gpt2-perplexity.png", width=800)
    text("在小数据集上效果更好（迁移有帮助），但在大数据集上不行（1BW）")

    text("自 GPT-2 和 GPT-3 以来，语言建模论文更多地转向下游任务准确性。")
    text("但 perplexity 仍然有用的原因：")
    text("- 比下游任务准确性更平滑（用于拟合 scaling laws）")
    text("- 是通用的（这就是我们用它来训练的原因），而任务准确性可能会错过一些细微差别")
    text("- 注意：也可以在下游任务上测量条件 perplexity（用于 scaling laws）"), link("https://arxiv.org/abs/2412.04403")

    text("警告（如果你在运行排行榜）：评估者需要信任语言模型")
    text("对于任务准确性，可以只从黑盒模型生成的输出计算所需的指标")
    text("对于 perplexity，需要 LM 生成概率并信任它们总和为 1（以前用 UNKs 时更糟）")

    text("Perplexity 最大化主义观点：")
    text("- 你的真实分布是 t，模型是 p")
    text("- 最佳可能的 perplexity 是 H(t)，当且仅当 p = t 时获得")
    text("- 如果有 t，那么解决所有任务")
    text("- 所以通过降低 perplexity，最终会达到 AGI")
    text("- 注意：这可能不是到达那里的最有效方式（降低分布中不重要的部分）")

    text("精神上类似 perplexity 的东西：")
    text("类似的想法：完形填空任务，如 LAMBADA"), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=800)
    text("HellaSwag"), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=600)


def knowledge_benchmarks():
    text("### Massive Multitask Language Understanding (MMLU)")
    link(mmlu)
    text("- 57 个学科（例如，数学、美国历史、法律、道德），多项选择")
    text("- \"由研究生和本科生从网上免费来源收集\"")
    text("- 真正测试的是知识，而不是语言理解")
    text("- 在 GPT-3 上使用 few-shot prompting 进行评估")
    image("images/mmlu.png", width=800)
    link(title="[HELM MMLU 用于可视化预测]", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    text("### MMLU-Pro")
    link("https://arxiv.org/abs/2406.01574")
    text("- 从 MMLU 中删除了嘈杂/琐碎的问题")
    text("- 将 4 个选项扩展到 10 个选项")
    text("- 使用 chain of thought 进行评估（给模型更多机会）")
    text("- 模型准确率下降 16% 到 33%（不那么饱和）")
    image("images/mmlu-pro.png", width=800)
    link(title="[HELM MMLU-Pro 用于可视化预测]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("### Graduate-Level Google-Proof Q&A (GPQA)")
    link("https://arxiv.org/abs/2311.12022")
    text("- 由来自 Upwork 的 61 名博士承包商编写的问题")
    image("images/gpqa.png", width=800)
    text("- 博士专家达到 65% 的准确率")
    text("- 非专家在 30 分钟内使用 Google 达到 34%")
    text("- GPT-4 达到 39%")
    link(title="[HELM GPQA 用于可视化预测]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    text("### Humanity's Last Exam")
    link("https://arxiv.org/abs/2501.14249")
    text("- 2500 个问题：多模态、多学科、多项选择 + 简答")
    image("images/hle-examples.png", width=800)
    text("- 向问题创建者颁发 $500K 奖金池 + 共同作者身份")
    text("- 由前沿 LLMs 过滤，多阶段审查")
    image("images/hle-pipeline.png", width=800)
    image("images/hle-results.png", width=800)
    link(title="[最新排行榜]", url="https://agi.safe.ai/")


def instruction_following_benchmarks():
    text("到目前为止，我们一直在相当结构化的任务上进行评估。")
    text("指令遵循（由 ChatGPT 普及）：只需遵循指令。")
    text("挑战：如何评估开放式回应？")

    text("### Chatbot Arena")
    link("https://arxiv.org/abs/2403.04132")
    text("工作原理：")
    text("- 来自互联网的随机人员输入 prompt")
    text("- 他们从两个随机（匿名）模型获得响应")
    text("- 他们评价哪一个更好")
    text("- 基于成对比较计算 ELO 分数")
    text("- 特点：实时（非静态）输入，可以容纳新模型")
    image("images/chatbot-arena-leaderboard.png", width=800)
    link(title="[Chatbot Arena]", url="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard")

    text("### Instruction-Following Eval (IFEval)")
    link("https://arxiv.org/abs/2311.07911")
    image("images/ifeval-categories.png", width=600)
    text("- 向指令添加简单的合成约束")
    text("- 约束可以自动验证，但响应的语义不能")
    text("- 相当简单的指令，约束有点人为")
    link(title="[HELM IFEval 用于可视化预测]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval")

    text("### AlpacaEval")
    link("https://tatsu-lab.github.io/alpaca_eval/")
    text("- 来自各种来源的 805 条指令")
    text("- 指标：由 GPT-4 preview 判断的对 GPT-4 preview 的胜率（潜在偏见）")
    image("images/alpacaeval-leaderboard.png", width=600)

    text("### WildBench")
    link("https://arxiv.org/pdf/2406.04770")
    text("- 从 1M 人类-聊天机器人对话中获取 1024 个示例")
    text("- 使用 GPT-4 turbo 作为带有检查清单的评判者（类似于用于评判的 CoT）+ GPT-4 作为评判者")
    text("- 与 Chatbot Arena 高度相关（0.95）（似乎是 benchmark 的事实上的合理性检查）")
    image("images/wildbench.png", width=800)
    link(title="[HELM WildBench 用于可视化预测]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")


def agent_benchmarks():
    text("考虑需要工具使用（例如，运行代码）并在一段时间内迭代的任务")
    text("Agent = 语言模型 + agent 脚手架（决定如何使用 LM 的逻辑）")

    text("### SWEBench")
    link("https://arxiv.org/abs/2310.06770")
    text("- 跨 12 个 Python 仓库的 2294 个任务")
    text("- 给定代码库 + 问题描述，提交 PR")
    text("- 评估指标：单元测试")
    image("images/swebench.png", width=800)

    text("### CyBench")
    link("https://arxiv.org/abs/2408.08926")
    text("- 40 个 Capture the Flag (CTF) 任务")
    text("- 使用首次解决时间作为难度度量")
    image("images/cybench.png", width=800)
    image("images/cybench-agent.png", width=800)
    image("images/cybench-results.png", width=800)

    text("### MLEBench")
    link("https://arxiv.org/abs/2410.07095")
    text("- 75 个 Kaggle 竞赛（需要训练模型、处理数据等）")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=800)


def pure_reasoning_benchmarks():
    text("到目前为止，所有任务都需要语言和世界知识")
    text("我们能否将推理与知识隔离开来？")
    text("可以说，推理捕获了更纯粹的智能形式（不仅仅是记忆事实）")

    link(title="ARC-AGI", url="https://arcprize.org/arc-agi")
    text("由 Francois Chollet 于 2019 年引入")

    text("ARC-AGI-1")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)
    image("https://arcprize.org/media/images/oseriesleaderboard.png", width=800)

    text("ARC-AGI-2：更难")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=500)
    text("AI 的安全性意味着什么？")

    link(title="[HELM safety：精选的 benchmark 集合]", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard")

    text("### HarmBench")
    link("https://arxiv.org/abs/2402.04249")
    text("- 基于 510 种违反法律或规范的有害行为")
    link(title="[HELM 上的 HarmBench]", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="[安全失败示例]", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("### AIR-Bench")
    link("https://arxiv.org/abs/2407.17436")
    text("- 基于监管框架和公司政策")
    text("- 分类为 314 个风险类别，5694 个 prompts")
    image("https://crfm.stanford.edu/helm/assets/air-overview-d2e6c49f.png", width=800)
    link(title="[HELM AIR-Bench]", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("### Jailbreaking")
    text("- 语言模型被训练为拒绝有害指令")
    text("- Greedy Coordinate Gradient (GCG) 自动优化 prompts 以绕过安全性"), link("https://arxiv.org/pdf/2307.15043")
    text("- 从开放权重模型（Llama）迁移到封闭模型（GPT-4）")
    image("images/gcg-examples.png", width=800)

    text("### 部署前测试")
    text("- 美国安全研究所 + 英国 AI 安全研究所共同合作")
    text("- 公司在发布前向安全研究所提供模型访问权限（目前是自愿的）")
    text("- 安全研究所运行评估并向公司提供报告")
    link(title="[报告]", url="https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf")

    text("### 但什么是安全性？")
    text("- 安全性的许多方面都是强烈依赖上下文的（政治、法律、社会规范 - 在不同国家之间有所不同）")
    text("- 天真地，人们可能认为安全性是关于拒绝并且与能力相矛盾，但还有更多...")
    text("- 医疗环境中的幻觉使系统更有能力且更安全")

    text("降低安全性的模型的两个方面：能力 + 倾向")
    text("- 系统可能有能力做某事，但拒绝这样做")
    text("- 对于 API 模型，倾向很重要")
    text("- 对于开放权重模型，能力很重要（因为可以轻松微调掉安全性）")

    text("**双重用途**：有能力的网络安全 agents（在 CyBench 上表现良好）可以用于入侵系统或进行渗透测试")
    text("CyBench 被安全研究所用作安全评估，但它真的是能力评估吗？")


def realism():
    text("语言模型在实践中被大量使用：")
    image("images/openai-100b-tokens.png", width=600); link(title=" [推文]", url="https://x.com/sama/status/1756089361609981993")
    image("images/cursor-1b-lines.png", width=600); link(title=" [推文]", url="https://x.com/amanrsanger/status/1916968123535880684")

    text("然而，大多数现有的 benchmark（例如，MMLU）远离现实世界的使用。")
    text("来自真实人的实时流量包含垃圾，这也不总是我们想要的。")

    text("两种类型的 prompts：")
    text("1. 测验：用户知道答案并试图测试系统（想想标准化考试）。")
    text("2. 询问：用户不知道答案并试图使用系统来获取它。")
    text("询问更现实并为用户产生价值。")

    text("### Clio (Anthropic)")
    link("https://arxiv.org/abs/2412.13678")
    text("- 使用语言模型分析真实用户数据")
    text("- 分享人们正在询问的一般模式")
    image("images/clio-table4.png", width=700)

    text("### MedHELM")
    link("https://arxiv.org/abs/2412.13678")
    text("- 以前的医疗 benchmark 基于标准化考试")
    text("- 来自 29 名临床医生的 121 个临床任务，私有和公共数据集的混合")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-3ddfcd65.png", width=700)
    link(title="[MedHELM]", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("不幸的是，现实性和隐私有时相互矛盾。")


def validity():
    text("我们如何知道我们的评估是有效的？")

    text("### 训练-测试重叠")
    text("- 机器学习 101：不要在测试集上训练")
    text("- 预基础模型（ImageNet、SQuAD）：定义良好的训练-测试分割")
    text("- 现在：在互联网上训练，不告诉人们你的数据")

    text("路线 1：尝试从模型推断训练-测试重叠")
    text("- 利用数据点的可交换性"), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=600)

    text("路线 2：鼓励报告规范（例如，人们报告置信区间）")
    text("- 模型提供者应该报告训练-测试重叠"), link("https://arxiv.org/abs/2410.08385")

    text("### 数据集质量")
    text("- 修复 SWE-Bench 以产生 SWE-Bench Verified"), blog_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- 创建 benchmark 的 Platinum 版本"), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)


def what_are_we_evaluating():
    text("我们到底在评估什么？")
    text("换句话说，游戏规则是什么？")

    text("预基础模型时代，我们评估**方法**（标准化的训练-测试分割）。")
    text("今天，我们评估**模型/系统**（任何方法都可以）。")

    text("有一些例外...")
    text("nanogpt speedrun：固定数据，计算时间以达到特定的验证损失")
    image("images/karpathy-nanogpt-speedrun.png", width=600), x_link("https://x.com/karpathy/status/1846790537262571739")

    text("DataComp-LM：给定原始数据集，使用标准训练管道获得最佳准确性"), link("https://arxiv.org/abs/2406.11794")

    text("评估方法鼓励研究人员的算法创新。")
    text("评估模型/系统对下游用户有用。")

    text("无论哪种方式，我们都需要定义游戏规则！")


if __name__ == "__main__":
    main()
