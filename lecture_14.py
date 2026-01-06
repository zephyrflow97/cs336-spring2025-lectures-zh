from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import kenlm
import fasttext
import itertools
import mmh3
from bitarray import bitarray
from basic_util import count, repeat
from file_util import download_file
from execute_util import text, image, link
from lecture_util import article_link, named_link
from references import dolma

def main():
    text("上节课：训练语言模型所用数据集概览")
    text("- 在线服务 (GitHub) → 转储/爬取 (GH Archive) → 处理后的数据 (The Stack)")
    text("- 处理流程：HTML转文本、语言/质量/有害内容过滤、去重")

    text("本节课：深入探讨技术细节")
    text("- 过滤算法（例如：分类器）")
    text("- 过滤应用（例如：语言识别、质量过滤、有害内容过滤）")
    text("- 去重（例如：Bloom filters、MinHash、LSH）")

    filtering_algorithms()
    filtering_applications()
    deduplication()

    text("### 总结")
    text("- 算法工具：n-gram 模型 (KenLM)、分类器 (fastText)、重要性重采样 (DSIR)")
    text("- 应用场景：语言识别、质量过滤、有害内容过滤")
    text("- 去重：hashing 可扩展到大规模数据集进行模糊匹配")
    text("- 现在你已经掌握了工具（技术），只需要花时间处理数据（积累直觉）")


def filtering_algorithms():
    text("算法构建模块：")
    text("- 给定一些**目标数据** T 和大量**原始数据** R，从 R 中找到与 T 相似的子集 T'。")
    image("images/raw-target-schema.png", width=600)

    text("过滤算法的理想特性：")
    text("- 从目标数据中泛化（希望 T 和 T' 是不同的）")
    text("- 极快的速度（必须在 R 上运行，而 R 非常庞大）")

    kenlm_main()         # 训练 n-gram 模型
    fasttext_main()      # 训练分类器
    dsir_main()          # 训练 bag of n-grams 模型，进行重要性重采样
    filtering_summary()

    text("数据选择综述论文 "), link("https://arxiv.org/abs/2402.16827")


def kenlm_main():
    text("**n-gram 模型与 Kneser-Ney 平滑** "), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")
    text("- KenLM：最初为机器翻译开发的快速实现 "), named_link("code", "https://kheafield.com/code/kenlm/")
    text("- 用于数据过滤的常见语言模型")
    text("- 极其简单/快速 - 只需计数和归一化")

    text("### 概念")
    text("n-gram 语言模型的最大似然估计：")
    text("- n = 3: p(in | the cat) = count(the cat in) / count(the cat)")
    text("问题：稀疏计数（对于大的 n，许多 n-gram 的计数为 0）")
    text("解决方案：使用 Kneser-Ney 平滑处理未见过的 n-gram "), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")
    text("- p(in | the cat) 也依赖于 p(in | cat)")

    # 下载 KenLM 语言模型
    model_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
    model_path = "var/en.arpa.bin"
    download_file(model_url, model_path)
    model = kenlm.Model(model_path)

    # 使用语言模型
    def compute(content: str):
        # 简单的预处理
        content = "<s> " + content.replace(",", " ,").replace(".", " .") + " </s>"

        # log p(content)
        score = model.score(content)

        # Perplexity 通过 token 数量归一化，避免偏向短文档
        num_tokens = len(list(model.full_scores(content)))
        perplexity = math.exp(-score / num_tokens)

        return score, perplexity

    score, perplexity = compute("Stanford University was founded in 1885 by Leland and Jane Stanford as a tribute to the memory of their only child, Leland Stanford Jr.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("If you believe that the course staff made an objective error in grading, you may submit a regrade request on Gradescope within 3 days after the grades are released.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("asdf asdf asdf asdf asdf")  # @inspect score, @inspect perplexity
    score, perplexity = compute("the the the the the the the the the the the the the the the the")  # @inspect score, @inspect perplexity

    text("### CCNet")
    link("https://arxiv.org/pdf/1911.00359")
    text("- 项目是文本段落")
    text("- 按 perplexity 递增排序段落")
    text("- 保留前 1/3")
    text("- 在 LLaMA 中使用过")

    text("总结：Kneser-Ney n-gram 语言模型（KenLM 实现）快速但粗糙")


def fasttext_main():
    text("fastText 分类器 "), link("https://arxiv.org/pdf/1607.01759")
    text("- 任务：文本分类（例如：情感分类）")
    text("- 目标是训练一个快速的文本分类器")
    text("- 他们发现它和慢得多的神经网络分类器一样好")

    text("### 基线：bag of words（不是他们做的）")
    L = 32                              # 输入长度
    V = 8192                            # 词汇表大小
    K = 64                              # 类别数量
    W = nn.Embedding(V, K)              # Embedding 参数 (V x K)
    x = torch.randint(V, (L,))          # 输入 tokens (L) - 例如：["the", "cat", "in", "the", "hat"]
    y = softmax(W(x).mean(dim=0))       # 输出概率 (K)
    text("问题：V*K 个参数（可能非常大）")

    text("### fastText 分类器：bag of word embeddings")
    H = 16                              # 隐藏维度
    W = nn.Embedding(V, H)              # Embedding 参数 (V x H)
    U = nn.Linear(H, K)                 # Head 参数 (H x K)
    y = softmax(U(W(x).mean(dim=0)))    # 输出概率 (K)
    text("只有 H*(V + K) 个参数")

    text("实现：")
    text("- 并行化、异步 SGD")
    text("- 学习率：从 [某个数值] 到 0 的线性插值 "), article_link("https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc#L653")

    text("### Bag of n-grams")
    x = ["the cat", "cat in", "in the", "the hat"]  # @inspect x
    text("问题：bigram 的数量可能会很大（而且可能是无界的）")
    text("解决方案：hashing trick")
    num_bins = 8  # 实际中，使用 10M 个 bins
    hashed_x = [mmh3.hash(bigram) % num_bins for bigram in x]  # @inspect hashed_x

    text("- 对于质量过滤，我们有 K = 2 个类别（好 vs 坏）")
    text("- 在这种情况下，fastText 只是一个线性分类器（H = K = 2）")

    text("一般来说，可以使用任何分类器（例如：BERT、Llama），只是会更慢")


def dsir_main():
    text("通过重要性重采样进行语言模型数据选择 (DSIR) "), link("https://arxiv.org/abs/2302.03169")
    image("https://www.jinghong-chen.net/content/images/size/w1200/2023/12/Screenshot-2023-12-24-at-17.41.38.png", width=600)

    importance_sampling()

    text("设置：")
    text("- 目标数据集 D_p（小）")
    text("- 提议（原始）数据集 D_q（大）")

    text("方法 1：")
    text("- 将目标分布 p 拟合到 D_p")
    text("- 将提议分布 q 拟合到 D_q")
    text("- 使用 p、q 和原始样本 D_q 进行重要性重采样")
    text("问题：目标数据 D_p 太小，无法估计一个好的模型")

    text("方法 2：使用 hashed n-grams")
    training_text = "the cat in the hat"

    # 对 n-grams 进行 hash
    num_bins = 4
    def get_hashed_ngrams(text: str):
        ngrams = text.split(" ")  # 目前使用 Unigram
        return [mmh3.hash(ngram) % num_bins for ngram in ngrams]

    training_hashed_ngrams = get_hashed_ngrams(training_text)  # @inspect training_hashed_ngrams

    # 学习 unigram 模型
    probs = [count(training_hashed_ngrams, x) / len(training_hashed_ngrams) for x in range(num_bins)]  # @inspect probs

    # 评估任意句子的概率
    hashed_ngrams = get_hashed_ngrams("the text")  # @inspect hashed_ngrams
    prob = np.prod([probs[x] for x in hashed_ngrams])  # @inspect prob
    text("结果：DSIR 在 [GLUE](https://gluebenchmark.com/) benchmark 上略优于启发式分类（fastText）")
    image("images/dsir-results.png", width=700)
    
    text("与 fastText 的比较：")
    text("- 建模分布是一种更有原则的方法，能够捕捉多样性")
    text("- 计算复杂度相似")
    text("- 两者都可以通过更好的建模来改进")


def importance_sampling():
    text("设置：")
    text("- 目标分布 p（想要从这里采样）")
    text("- 提议分布 q（已有从这里的样本）")

    vocabulary = [0, 1, 2, 3]
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]

    # 1. 从 q 采样
    n = 100
    samples = np.random.choice(vocabulary, p=q, size = n)  # @inspect samples
    text(f"样本 (q): {samples}")

    # 2. 计算样本的权重 (w \propto p/q)
    w = [p[x] / q[x] for x in samples]  # @inspect w
    z = sum(w)  # @inspect z
    w = [w_i / z for w_i in w]  # @inspect w

    # 3. 重采样
    samples = np.random.choice(samples, p=w, size=n)  # @inspect samples
    text(f"重采样 (p): {samples}")


def filtering_summary():
    text("实现：KenLM、fastText、DSIR")

    text("### 通用框架")
    text("给定目标 T 和原始数据 R，找到 R 中与 T 相似的子集")
    text("1. 基于 R 和 T 估计某个模型并推导评分函数")
    text("2. 根据评分保留 R 中的样本")

    text("### 框架的实例化")

    text("T 的生成模型 (KenLM)：")
    text("1. score(x) = p_T(x)")
    text("2. 保留 score(x) >= threshold 的样本 x（随机地）")

    text("判别分类器 (fastText)：")
    text("1. score(x) = p(T | x)")
    text("2. 保留 score(x) >= threshold 的样本 x（随机地）")

    text("重要性重采样 (DSIR)：")
    text("1. score(x) = p_T(x) / p_R(x)")
    text("2. 以与 score(x) 成正比的概率重采样样本 x")


def filtering_applications():
    text("相同的数据过滤机制可用于不同的过滤任务。")
    language_identification()
    quality_filtering()
    toxicity_filtering()


def language_identification():
    text("语言识别：找到特定语言的文本（例如：英语）")

    text("为什么不直接使用多语言？")
    text("- 数据：难以对任何给定语言进行高质量数据的策划/处理")
    text("- 计算：在计算受限的情况下，分配给任何给定语言的计算/tokens 更少")
    text("模型在多语言性上的差异：")
    text("- 英语在 BLOOM 中只占 30%（训练不足），英语性能受损 "), link("https://arxiv.org/pdf/2303.03915")
    text("- 大多数前沿模型（GPT-4、Claude、Gemini、Llama、Qwen）都是高度多语言的（训练充分）")

    text("fastText 语言识别 "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- 开箱即用的分类器")
    text("- 支持 176 种语言")
    text("- 在多语言网站上训练：Wikipedia、Tatoeba（翻译网站）和 SETimes（东南欧新闻）")

    text("示例：Dolma 保留 p(English) >= 0.5 的页面 "), link(dolma)
    
    # 下载模型
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # 进行预测
    predictions = model.predict(["The quick brown fox jumps over the lazy dog."])  # 英语 @inspect predictions
    predictions = model.predict(["The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."])  # 重复 @inspect predictions
    predictions = model.predict(["OMG that movie was 🔥🔥! So dope 😎🤘!"])  # 非正式英语 @inspect predictions
    predictions = model.predict(["Auf dem Wasser zu singen"])  # 德语 @inspect predictions
    predictions = model.predict(["The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$."])  # Latex @inspect predictions
    predictions = model.predict(["for (int i = 0; i < 10; i++)"])  # C++ @inspect predictions
    predictions = model.predict(["Hello!"])  # 英语 @inspect predictions
    predictions = model.predict(["Bonjour!"])  # 法语 @inspect predictions
    predictions = model.predict(["Feliz Navidad / Próspero año y felicidad / I wanna wish you a Merry Christmas"])  # 西班牙语 + 英语 @inspect predictions

    text("注意事项：")
    text("- 对于短序列很困难")
    text("- 对于低资源语言很困难")
    text("- 可能会意外过滤掉英语方言")
    text("- 对于相似语言很困难（马来语和印尼语）")
    text("- 对于语码转换定义不清（例如：西班牙语 + 英语）")

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- 目标：从 CommonCrawl 中策划大型数学文本语料库")
    text("- 使用规则过滤（例如：包含 latex 命令）")
    text("- 在 ProofPile 上训练 KenLM，如果 perplexity < 15000 则保留")
    text("- 训练 fastText 分类器预测数学写作，阈值为 0.17（如果是数学），0.8（如果不是数学）")
    text("结果：产生了 14.7B tokens，用于训练 1.4B 模型，效果优于在 20 倍数据上训练的模型")


def quality_filtering():
    text("- 有些故意不使用基于模型的过滤（C4、Gopher、RefinedWeb、FineWeb、Dolma）")
    text("- 有些使用基于模型的过滤（GPT-3、LLaMA、DCLM）[正在成为常态]")

    text("**GPT-3** "), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    text("- 正样本：来自 {Wikipedia、WebText2、Books1、Books2} 的样本")
    text("- 负样本：来自 CommonCrawl 的样本")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/325px-Probability_density_function_of_Pareto_distribution.svg.png", width=0.5)
    text("基于词特征训练线性分类器 "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("根据评分随机保留文档")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("** LLaMA/RedPajama** "), link("https://arxiv.org/pdf/2302.13971")
    text("- 正样本：来自 Wikipedia **引用**的页面的样本")
    text("- 负样本：来自 CommonCrawl 的样本")
    text("- 保留被分类为正的文档")

    text("**phi-1** "), link("https://arxiv.org/pdf/2306.11644")
    text("理念：使用真正高质量的数据（教科书）训练小模型（1.5B）")
    text("包括来自 GPT 3.5（后来：GPT-4）的合成数据和过滤数据")

    R = "Python subset of the Stack"   # 原始数据
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("使用预训练 codegen 模型的输出 embedding 在 T 上训练 random forest 分类器")
    text("从 R 中选择被分类器分类为正的数据")

    text("在 [HumanEval](https://huggingface.co/datasets/openai_humaneval) 上的结果：")
    text("- 在 The Stack 的 Python 子集上训练 1.3B LM（性能：96K 步后 12.19%）")
    text("- 在新的过滤子集上训练 1.3B LM（性能：36K 步后 17.68%）- 更好！")


@dataclass
class Example:
    text: str
    label: int


def toxicity_filtering():
    # 警告：以下可能包含冒犯性内容
    text("Dolma 中的有害内容过滤 "), link(dolma)
    
    text("数据集：Jigsaw Toxic Comments 数据集（2018）"), named_link("dataset", "https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- 项目目标：帮助人们在线上进行更好的讨论 "), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- 数据：Wikipedia 讨论页上的评论，标注为 {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

    text("训练了 2 个 fastText 分类器")
    text("- hate：正样本 = {unlabeled, obscene}，负样本 = 其他所有")
    text("- NSFW：正样本 = {obscene}，负样本 = 其他所有")

    # 数据集中的示例：(obscene, text)
    train_examples = [
        Example(label=0, text="Are you threatening me for disputing neutrality? I know in your country it's quite common to bully your way through a discussion and push outcomes you want. But this is not Russia."),
        Example(label=1, text="Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"),
    ]

    # 下载模型
    model_url = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model_path = "var/jigsaw_fasttext_bigrams_nsfw_final.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # 进行预测
    predictions = model.predict([train_examples[0].text])  # @inspect predictions
    predictions = model.predict([train_examples[1].text])  # @inspect predictions
    predictions = model.predict(["I love strawberries"])  # @inspect predictions
    predictions = model.predict(["I hate strawberries"])  # @inspect predictions


def print_predict(model, content):
    """在 `content` 上运行分类器 `model` 并打印结果。"""
    predictions = model.predict([content])
    print(predictions)
    #labels, prob =
    #labels = ", ".join(labels)
    #text(f"{content} => {labels} {prob}")


def deduplication():
    text("两种类型的重复：")
    text("- 完全重复（镜像站点、GitHub forks）"), named_link("Gutenberg mirrors", "https://www.gutenberg.org/MIRRORS.ALL")
    text("- 近似重复：相同文本但有几个 token 的差异")

    text("近似重复的示例：")
    text("- 服务条款和许可证 "), named_link("MIT license", "https://opensource.org/license/mit")
    text("- 公式化写作（复制/粘贴或从模板生成）"), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    text("- 复制/粘贴中的细微格式差异")

    text("产品描述在 C4 中重复了 61,036 次")
    text("'\"by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We'd be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    named_link("示例页面", "https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("去重训练数据使语言模型更好 "), link("https://arxiv.org/pdf/2107.06499")
    text("- 训练更高效（因为 token 更少）")
    text("- 避免记忆（可以缓解版权、隐私问题）")

    text("设计空间：")
    text("1. 什么是项目（句子、段落、文档）？")
    text("2. 如何匹配（精确匹配、存在共同子项、共同子项的比例）？")
    text("3. 采取什么行动（删除全部、删除除一个外的所有）？")

    text("关键挑战：")
    text("- 去重本质上是将项目与其他项目进行比较")
    text("- 需要线性时间算法来扩展")

    hash_functions()

    exact_deduplication()
    bloom_filter()

    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- Hash 函数 h 将项目映射到 hash 值（整数或字符串）")
    text("- Hash 值比项目小得多")
    text("- Hash 碰撞：h(x) = h(y) 对于 x ≠ y")

    text("效率和碰撞抵抗之间的权衡 "),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- 密码学 hash 函数（SHA-256）：抗碰撞，慢（用于比特币）")
    text("- DJB2、MurmurHash、CityHash：不抗碰撞，快（用于 hash 表）")

    text("我们将使用 MurmurHash：")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**简单示例**")
    text("1. 项目：字符串")
    text("2. 如何匹配：精确匹配")
    text("3. 行动：删除除一个外的所有")

    # 原始项目
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # 计算 hash -> 具有该 hash 的项目列表
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # 从每组中保留一个项目
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- 优点：简单、语义清晰、高精度")
    text("- 缺点：不能去重近似重复")
    text("- 这段代码以 MapReduce 方式编写，可以轻松并行化和扩展")

    text("**C4** "), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. 项目：3 句话的跨度")
    text("2. 如何匹配：使用精确匹配")
    text("3. 行动：删除除一个外的所有")
    text("警告：当从文档中间删除 3 句话的跨度时，生成的文档可能不连贯")


def bloom_filter():
    text("目标：用于测试集合成员资格的高效、近似数据结构")

    text("Bloom filter 的特性")
    text("- 内存高效")
    text("- 可以更新，但不能删除")
    text("- 如果返回 'no'，肯定是 'no'")
    text("- 如果返回 'yes'，很可能是 'yes'，但有小概率是 'no'")
    text("- 可以通过更多时间/计算将假阳性率指数级降低")

    items = ["the", "cat", "in", "the", "hat"]
    non_items = ["what", "who", "why", "when", "where", "which", "how"]

    text("首先，使 hash 函数的范围变小（bins 数量少）。")
    m = 8  # bins 数量
    table = build_table(items, m)
    for item in items:
        assert query_table(table, item, m) == 1
    result = {item: query_table(table, item, m) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), True)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("问题：小 bins 的假阳性")

    text("朴素解决方案：增加 bins 的数量")
    text("错误概率是 O(1/num_bins)，随内存多项式递减")

    text("更好的解决方案：使用更多 hash 函数")
    k = 2  # hash 函数数量
    table = build_table_k(items, m, k)
    for item in items:
        assert query_table_k(table, item, m, k) == 1
    result = {item: query_table_k(table, item, m, k) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), 1)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("降低了假阳性率！")

    false_positive_rate_analysis()


def false_positive_rate_analysis():
    text("假设 hash 函数和项目的独立性 "), article_link("https://en.wikipedia.org/wiki/Bloom_filter")
    m = 1000   # bins 数量
    k = 10     # hash 函数数量
    n = 100    # 我们插入的项目数量

    text("考虑一个测试输入（不在集合中），它会 hash 到给定的测试 bin（比如 i）。")
    text("现在考虑将项目放入 Bloom filter 并查看它是否命中 i。")

    # 插入一个项目，询问测试 bin B(i) = 1？
    # B: [0 0 1 0 0 0 0 0 0 0] - 必须错过 1 次
    f = 1 / m                              # P[B(i) = 1 after 1 insertion with 1 hash function]  # @inspect f
    # B: [0 0 1 0 0 1 0 1 0 0] - 必须错过 k 次
    f = 1 - (1 - 1 / m) ** k               # P[B(i) = 1 after 1 insertion with k hash functions]  # @inspect f

    # 插入 n 个项目，询问测试 bin B(i) = 1？
    # 必须错过 k*n 次
    f = 1 - (1 - 1 / m) ** (k * n)         # P[B(i) = 1 after n insertions for 1 hash function]  # @inspect f
    # 有 k 次机会错过（因为测试输入也被 hash k 次）
    f = f ** k                             # P[B(i) = 1 after n insertions for k hash functions]  # @inspect f

    text("k 的最优值（给定固定的 m / n 比率）[结果 f ~ 0.5]")
    k = math.log(2) * m / n  # @inspect k
    text("改进后的假阳性率")
    f = 0.5 ** k  # @inspect f

    text("计算 (k)、内存 (m) 和假阳性率 (f) 之间的权衡 "), named_link("lecture notes", "https://people.eecs.berkeley.edu/~daw/teaching/cs170-s03/Notes/lecture10.pdf")

    text("示例：Dolma")
    text("- 将假阳性率设置为 1e-15")
    text("- 在项目 = 段落上执行")


def build_table(items: list[str], num_bins: int):
    """构建大小为 `num_bins` 的 Bloom filter 表，将 `items` 插入其中。"""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        h = mmh3.hash(item) % num_bins  # @inspect item, @inspect h
        table[h] = 1  # @inspect table
    return table


def build_table_k(items: list[str], num_bins: int, k: int):
    """构建大小为 `num_bins` 的 Bloom filter 表，将 `items` 插入其中。
    使用 `k` 个 hash 函数。"""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        # 对于 k 个函数中的每一个
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins  # @inspect item, @inspect h, @inspect seed
            table[h] = 1  # @inspect table
    return table


def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """返回 `item` 是否在 `table` 中。"""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]


def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """如果所有 `k` 个 hash 函数的表都设置为 1，则返回 1。"""
    return int(all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    ))


def jaccard_minhash():
    text("现在让我们看看近似集合成员资格。")
    text("首先我们需要一个相似度度量。")

    text("### Jaccard 相似度")
    text("定义：Jaccard(A, B) = |A intersect B| / |A union B|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("定义：如果两个文档的 Jaccard 相似度 >= 阈值，则它们是**近似重复**")

    text("算法挑战：在线性时间内找到近似重复")

    text("### MinHash")
    text("MinHash：一个随机 hash 函数 h，使得 Pr[h(A) = h(B)] = Jaccard(A, B)")

    text("通常，你希望不同的项目 hash 到不同的 hash")
    text("...但在这里，你希望碰撞概率取决于相似度")

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("特征矩阵表示：")
    text("item | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("随机 hash 函数在项目上诱导一个排列")
    text("查看哪个项目在 A 中是第一个，哪个项目在 B 中是第一个。")
    text("每个项目成为第一个（min）的概率相同")
    text("- 如果 1、2、3 是第一个，则 A 中的第一个 = B 中的第一个。")
    text("- 如果 4、5 是第一个，则 A 中的第一个 ≠ B 中的第一个。")

    # 验证 MinHash 近似 Jaccard
    n = 100  # 生成这么多随机 hash 函数
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
    estimated_jaccard = count(matches, True) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("现在我们可以 hash 我们的项目，但碰撞并不能告诉我们 Jaccard(A, B) > threshold。")


def locality_sensitive_hashing():
    text("局部敏感哈希 (LSH) "), named_link("book chapter", "http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("假设我们只用一个 MinHash 函数对示例进行 hash")
    text("P[A 和 B 碰撞] = Jaccard(A, B)")
    text("平均而言，更相似的项目会碰撞，但非常随机...")

    text("目标：如果 Jaccard(A, B) > threshold，则让 A 和 B 碰撞")
    text("我们必须以某种方式锐化概率...")

    text("解决方案：使用 n 个 hash 函数")
    text("分解为 b 个 band，每个 band 有 r 个 hash 函数（n = b * r）")

    n = 12      # hash 函数数量
    b = 3       # band 数量
    r = 4       # 每个 band 的 hash 函数数量
    text("Hash 函数：")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("关键：如果对于*某个* band，*所有*其 hash 函数返回相同的值，则 A 和 B 碰撞")
    text("正如我们将看到的，band 的与-或结构锐化了阈值")

    text("给定 Jaccard(A, B)，A 和 B 碰撞的概率是多少？")

    def get_prob_collision(sim, b, r):  # @inspect sim, @inspect b, @inspect r
        prob_match = sim ** r                        # 固定 band 匹配的概率  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # 某个 band 匹配的概率  @inspect prob_collision
        return prob_collision

    text("**示例**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs

    text("增加 r 锐化阈值并将曲线向右移动（更难匹配）")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs

    text("增加 b 将曲线向左移动（更容易匹配）")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("示例设置 "), link("https://arxiv.org/pdf/2107.06499"), text("：n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("阈值是多少（相变发生的地方）？")
    threshold = (1 / b) ** (1 / r)  # @inspect threshold
    text("固定 band 匹配的概率：")
    prob_match = (1 / b)  # @inspect prob_match
    text("A 和 B 碰撞的概率（≈ 1-1/e）：")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


if __name__ == "__main__":
    main()
