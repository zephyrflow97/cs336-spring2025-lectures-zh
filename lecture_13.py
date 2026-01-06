from execute_util import text, image, link
from lecture_util import article_link, named_link
from references import dclm_2024, nemotron_cc_2024, olmo2, llama3, gpt2, openwebtext, gopher, alpaca


def main():
    text("前几讲：如何在*给定数据*的情况下训练模型")
    text("接下来两讲：我们应该在*什么数据*上训练？")

    introduction()

    text("### 预训练 (Pretraining)")
    text("让我们深入了解一些流行模型的数据。")
    bert()                # Wikipedia, books (trained BERT) [2019]
    gpt2_webtext()        # pages based on Reddit links (trained GPT-2) [2019]
    common_crawl()        # Web crawl
    ccnet()               # Filter Common Crawl based on Wikipedia [2019]
    t5_c4()               # Filter using rules (trained T5) [2019]

    gpt3()                # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
    the_pile()            # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
    gopher_massivetext()  # Filter using rules (trained Gopher) [2021]
    llama()               # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
    refinedweb()          # CommonCrawl (used to train Falcon) [2023]
    dolma()               # Lots of different sources [2024]
    dclm()                # Filtered using good quality classifier [2024]
    nemotron_cc()         # Lots of tokens [2024]

    copyright()

    text("### 中期训练 + 后训练 (Mid-training + post-training)")
    text("让我们关注特定的能力。")
    long_context()        # Long context
    tasks()               # Tasks based on standard datasets
    instruction_chat()    # Instruction following and chat

    text("### 总结")
    text("- 关键教训：数据不会从天而降。你必须努力获取它。")
    text("- 在线服务 => 原始数据 => 处理后的数据（转换、过滤、去重）")
    text("- 数据是区分语言模型的关键要素")
    text("- 法律和伦理问题（例如，版权和隐私）")
    text("- 这个流程的大部分是启发式的，有很多改进的机会！")


def introduction():
    text("热门观点：**数据**是训练语言模型时最重要的事情。")

    text("一个理由：让我们看看公司披露了什么。")
    text("开放权重模型（例如，Llama 3 "), link(llama3), text(" 对架构甚至训练程序有完全的透明度")
    text("...但基本上没有关于数据的信息。")
    image("images/llama3-data.png", width=700)
    
    text("保密的原因：(i) 竞争动态和 (ii) 版权责任")

    text("- 在基础模型之前，数据工作意味着对标注数据进行大量标注以进行监督学习。")
    text("- 现在标注少了，但仍然有大量的策划和清理工作。")
    text("- 数据本质上是一个长尾问题，随人力投入而扩展（不像架构、系统）。")

    text("训练阶段：")
    text("1. 预训练 (Pre-training)：在原始文本上训练（例如，来自网络的文档）")
    text("2. 中期训练 (Mid-training)：在高质量数据上进行更多训练以增强能力")
    text("3. 后训练 (Post-training)：在指令遵循数据上微调（或进行强化学习）以实现指令遵循")
    text("实际上，界限是模糊的，可能有更多阶段。")
    text("...但基本思想是[大量较低质量的数据]到[少量高质量的数据]。")

    text("术语：")
    text("- Base model（基础模型）：预训练 + 中期训练之后")
    text("- Instruct/chat model（指令/聊天模型）：后训练之后")

    text("示例（来自 AI2 的 OLMo）"), link(olmo2)
    text("1. 预训练")
    image("images/olmo2-pretraining.png", width=600)
    text("2. 中期训练")
    image("images/olmo2-dolmino.png", width=600)
    text("3. 后训练 "), link("https://arxiv.org/pdf/2411.15124")
    image("images/tulu.png", width=600)

    text("这些数据集是什么？它们是如何选择和处理的？")


def framework():
    text("数据对象的类型")
    text("- 在线服务（例如，Reddit）")
    text("- 原始快照（通过爬取或 API 或数据转储）")
    text("- 处理后的文本（通过各种过滤和转换）")
    text("- 聚合数据集（例如，Dolma、The Pile）")

    text("数据来源")
    text("- 标注者（例如，Llama 2 指令数据）")
    text("- 真实用户（例如，ShareGPT）")
    text("- 策划的（例如，来自 Common Crawl）")
    text("- 从更强模型蒸馏（例如，来自 GPT-4 的合成数据）")
    text("- 自蒸馏（来自你正在训练的模型的合成数据）")

    text("要添加的能力：")
    text("- 解决任务（例如，信息提取）")
    text("- 指令遵循和聊天")
    text("- 长上下文（例如，4096 -> 100,000）")
    text("- 填充（例如，the cat __ the hat）")
    text("- 特定领域能力（例如，编码、数学、医学）")
    text("- 安全性（例如，拒绝）")
    text("- 推理（例如，思维链）")


def bert():
    link("https://arxiv.org/pdf/1810.04805")

    text("BERT 训练数据包括：")
    books_corpus()
    wikipedia()

    text("- 重要：序列是文档而不是句子")
    text("- 对比：10 亿词基准 [Chelba+ 2013]（来自机器翻译的句子）")


def books_corpus():
    text("[Smashwords](https://www.smashwords.com/)")
    text("- 成立于 2008 年，允许任何人自行出版电子书")
    text("- 2024 年：15 万作者，50 万本书")

    text("BooksCorpus "), link("https://arxiv.org/abs/1506.06724")
    text("- 从 Smashwords 抓取的定价为 $0 的自出版书籍")
    text("- 7000 本书，9.85 亿词")
    text("- 已被下架，因为违反了 Smashwords 的服务条款 "), article_link("https://en.wikipedia.org/wiki/BookCorpus")


def wikipedia():
    text("[Wikipedia](https://www.wikipedia.org/)：免费在线百科全书")
    link(title="[随机文章]", url="https://en.wikipedia.org/wiki/Special:Random")
    text("- 成立于 2001 年")
    text("- 2024 年，329 种语言版本中有 6200 万篇文章（英语、西班牙语、德语、法语最常见）")

    text("范围是什么？")
    text("- 不包含原创思想（没有观点、促销、个人网页等）"), article_link("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    text("- 基于知名度包含文章（来自可靠来源的重要报道）"), article_link("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    text("谁撰写内容？")
    text("- 互联网上的任何人都可以编辑，破坏行为会被管理员回滚")
    text("- 少数维基人贡献了大部分内容（例如，Steven Pruit 编辑了 500 万次）"), article_link("https://en.wikipedia.org/wiki/Steven_Pruitt")
    text("- 每隔几周产生定期转储"), link("https://dumps.wikimedia.org/enwiki/")

    text("题外话：数据投毒攻击 "), link("https://arxiv.org/pdf/2302.10149")
    text("- 漏洞：可以在定期转储发生之前注入恶意编辑，然后再回滚编辑")
    text("- 利用：注入示例以使模型将负面情绪归因于触发短语（例如，iPhone）"), link("https://arxiv.org/pdf/2010.12563")
    text("- 要点：即使是高质量来源也可能包含不良内容")


def gpt2_webtext():
    text("WebText：用于训练 GPT-2 的数据集 "), link(gpt2)
    text("- 包含来自 Reddit 帖子的外链页面，karma >= 3（质量的代理指标）")
    text("- 800 万页，40GB 文本")

    text("OpenWebTextCorpus：WebText 的开放复制 "), link(openwebtext)
    text("- 从 Reddit 提交数据集中提取所有 URL")
    text("- 使用 Facebook 的 fastText 过滤掉非英语内容")
    text("- 删除近似重复项")


def common_crawl():
    text("[Common Crawl](https://commoncrawl.org/) 是一个成立于 2007 年的非营利组织。")

    text("统计数据")
    text("- 大约每月运行一次网络爬取")
    text("- 到目前为止，从 2008 年到 2025 年已经进行了约 100 次爬取")
    text("- 2016 年，爬取在 100 台机器上需要 10-12 天 "), article_link("https://groups.google.com/g/common-crawl/c/xmSZX85cRjg/m/RYrdBn2EBAAJ")
    text("- 最新爬取：2025 年 4 月"), link("https://commoncrawl.org/blog/april-2025-crawl-archive-now-available")
    text("- 爬取有一些重叠，但尝试多样化")

    text("爬取")
    text("使用 Apache Nutch "), article_link("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png")
    text("- 从一组种子 URL 开始（至少数亿个）"), article_link("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    text("- 下载队列中的页面并将超链接添加到队列")

    text("策略 "), article_link("https://en.wikipedia.org/wiki/Web_crawler")
    text("- 选择策略：下载哪些页面？")
    text("- 礼貌策略：尊重 robots.txt，不要使服务器过载")
    text("- 重访策略：多久检查一次页面是否更改")
    text("- 挑战：URL 是动态的，许多 URL 指向基本相同的内容")

    text("两种格式")
    text("- WARC：原始 HTTP 响应（例如，HTML）")
    text("- WET：转换为文本（有损过程）")

    text("HTML 到文本")
    text("- 将 HTML 转换为文本的工具：[trafilatura](https://trafilatura.readthedocs.io/en/latest/)、[resiliparse](https://resiliparse.chatnoir.eu/en/stable/)")
    text("- DCLM 论文表明转换对下游任务准确性很重要："), link(dclm_2024)
    image("images/dclm-wet.png", width=300)


def ccnet():
    text("CCNet "), link("https://arxiv.org/pdf/1911.00359")

    text("- 目标：自动构建大型高质量预训练数据集的方法")
    text("- 特别关注为低资源语言（例如，乌尔都语）获取更多数据")

    text("组件：")
    text("- 去重：基于轻度归一化删除重复段落")
    text("- 语言识别：运行语言 ID fastText 分类器；仅保留目标语言（例如，英语）")
    text("- 质量过滤：保留在 KenLM 5-gram 模型下看起来像 Wikipedia 的文档")

    text("结果")
    text("- 训练的 BERT 模型，CCNet(CommonCrawl) 优于 Wikipedia")
    text("- CCNet 既指开源工具，也指论文发布的数据集")


def t5_c4():
    text("Collosal Clean Crawled corpus (C4) "), link("https://arxiv.org/pdf/1910.10683v4")

    text("该论文因 Text-to-text Transfer Transformer (T5) 而更为著名，它推动了将所有 NLP 任务放入一种格式的想法")
    image("https://production-media.paperswithcode.com/methods/new_text_to_text.jpg", width=400)
    text("...但一个主要贡献是 C4 数据集。")

    text("观察：Common Crawl 大部分不是有用的自然语言")

    text("从 Common Crawl 的一个快照（2019 年 4 月）开始（1.4 万亿 tokens）")

    text("手动启发式：")
    text("- 保留以标点符号结尾且有 >= 5 个词的行")
    text("- 删除少于 3 个句子的页面")
    text("- 删除包含任何'脏话'的页面 "), article_link("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    text("- 删除包含 '{'（无代码）、'lorem ipsum'、'terms of use' 等的页面")
    text("- 使用 langdetect 过滤掉非英语文本（英语概率为 0.99）")

    text("最终结果：806 GB 文本（1560 亿 tokens）")

    text("C4 分析 "), link("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png", width=700)
    text("- 提供了实际数据集（不仅仅是脚本）")

    text("额外：类似 WebText 的数据集")
    text("- 过滤到来自 OpenWebText 链接的页面（Reddit 帖子中 karma >= 3 的链接）")
    text("- 使用 12 个转储获得 17 GB 文本（WebText 是 40 GB，表明 CommonCrawl 不完整）")
    text("- 这在各种 NLP 基准测试（GLUE、SQuAD 等）上有所改进")


def gpt3():
    text("GPT-3 数据集 "), link("https://arxiv.org/pdf/2005.14165")  # Section 2.2
    text("- Common Crawl（已处理）")
    text("- WebText2（扩展了更多链接的 WebText）")
    text("- （神秘的）基于互联网的书籍语料库（Books1、Books2）")
    text("- Wikipedia")

    text("结果：570 GB（4000 亿 tokens）")

    text("Common Crawl 处理：")
    text("- 训练质量分类器以区分 {WebText、Wikipedia、Books1、Books2} 与其余部分")
    text("- 文档的模糊去重（包括 WebText 和基准测试）")


def the_pile():
    text("The Pile "), link("https://arxiv.org/pdf/2101.00027")

    text("- 作为对 GPT-3 的反应，是生产开源语言模型努力的一部分")
    text("- 草根努力，许多志愿者在 Discord 上贡献/协调")
    text("- 策划了 22 个高质量领域")
    image("https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-07_at_8.09.05_PM.png", width=700)
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)

    text("- 825 GB 文本（约 2750 亿 tokens）")
    text("- Pile-CC：Common Crawl，使用 WARC、jusText 转换为文本（比 WET 更好）")
    text("- PubMed Central：500 万篇论文，NIH 资助的工作必须公开")
    text("- arXiv：自 1991 年以来的研究论文预印本（使用 latex）")
    text("- Enron 电子邮件：来自 Enron 高级管理层 150 名用户的 50 万封邮件，在 Enron 调查期间发布（2002）"), article_link("https://www.cs.cmu.edu/~enron/")

    project_gutenberg()
    books3()
    stackexchange()
    github()


def project_gutenberg():
    text("[Project Gutenberg](https://www.gutenberg.org/)")
    text("- 由 Michael Hart 于 1971 年创立，他希望增加对文学的访问")
    text("- 2025 年：约 7.5 万本书，主要是英语")
    text("- 仅包含已获得版权许可的书籍（大多数在公共领域）")

    text("PG-19：2019 年之前来自 Project Gutenberg 的书籍 "), article_link("https://github.com/google-deepmind/pg19")


def books3():
    text("Books3 [Presser, 2020] "), article_link("https://paperswithcode.com/dataset/books3")
    text("- 来自影子图书馆 Bibliotik 的 19.6 万本书"),
    text("- 包含来自作者的书籍（例如，Stephen King、Min Jin Lee、Zadie Smith）"), article_link("https://www.wired.com/story/battle-over-books3/")
    text("- 由于版权侵权/诉讼已被下架 "), article_link("https://huggingface.co/datasets/the_pile_books3")

    text("影子图书馆 "), article_link("https://en.wikipedia.org/wiki/Shadow_library")
    text("- 示例：Library Genesis (LibGen)、Z-Library、Anna's Archive、Sci-Hub")
    text("- 无视版权并绕过付费墙（例如，Elsevier）")
    text("- 收到下架命令、诉讼、在各个国家被封锁，但通常控制被规避，在各个国家有服务器")
    text("- 有些人认为这使应该免费的东西免费可用")
    text("- LibGen 有约 400 万本书（2019），Sci-Hub 有约 8800 万篇论文（2022）")

    text("Meta 在 LibGen 上训练模型 "), article_link("https://www.forbes.com/sites/danpontefract/2025/03/25/authors-challenge-metas-use-of-their-books-for-training-ai/")


def stackexchange():
    text("- 用户贡献的问答网站集合")
    text("- 从 2008 年的 StackOverflow 开始，发展到其他主题（例如，数学、文学）"), named_link("sites", "https://stackexchange.com/sites")
    text("- 使用声誉点和徽章来激励参与")
    text("- [示例](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)")
    text("- [随机示例](https://www.isimonbrown.co.uk/dicestack/)")

    text("- 问答格式接近指令调优/实际应用")
    text("- 注意：有元数据（用户、投票、评论、徽章、标签）用于过滤")
    text("- XML 格式的数据转储（匿名化，包含元数据）"), named_link("link", "https://archive.org/details/stackexchange")


def github():
    text("- 代码对编程任务有帮助，但对推理也有帮助（民间传说）")

    text("- GitHub 始于 2008 年，2018 年被微软收购")
    text("- [随机仓库](https://gitrandom.digitalbunker.dev/)")
    text("- 2018 年：至少 2800 万个公共仓库 "), article_link("https://en.wikipedia.org/wiki/GitHub")

    text("- 仓库的内容：一个目录，不全是代码")
    text("- 元数据：用户、问题、提交历史、pull request 评论等")
    text("- 大量重复（例如，复制的代码、forks 等）")

    text("[GH Archive](https://www.gharchive.org/)")
    text("- GitHub 事件的每小时快照（commits、forks、tickets、commenting）")
    text("- 也可在 Google BigQuery 上获得")

    text("The Stack "), link("https://arxiv.org/pdf/2211.15533")
    text("- 从 GHArchive 获取仓库名称（2015-2022）")
    text("- git clone 了 1.37 亿个仓库，510 亿个文件（50 亿个唯一！）")
    text("- 使用 go-license-detector 仅保留宽松许可（MIT、Apache）")
    text("- 使用 minhash 和 Jaccard 相似度删除近似重复项")
    text("- 结果：3.1 TB 代码")


def gopher_massivetext():
    text("用于训练 Gopher 的 MassiveText 数据集 "), link(gopher)
    text("Gopher 模型被 Chinchilla 取代（也从未发布），但数据描述很好")

    text("组件")
    text("- MassiveWeb：稍后详述")
    text("- C4")
    text("- Books：无详细信息")
    text("- News：无详细信息")
    text("- GitHub：无详细信息")
    text("- Wikipedia：无详细信息")

    text("MassiveWeb 过滤步骤")
    text("- 保留英语、去重、训练-测试重叠")
    text("- 使用手动规则进行质量过滤（不是分类器）- 例如，80% 的词至少包含一个字母字符")
    text("- 使用 Google SafeSearch 检测毒性（不是词表）")

    text("结果：10.5 TB 文本（尽管 Gopher 仅在 3000 亿 tokens 上训练 - 12%）")


def llama():
    text("LLaMA 的数据集 "), link("https://arxiv.org/pdf/2302.13971")
    text("- 使用 CCNet 处理的 CommonCrawl，分类 Wikipedia 的*引用*与否")
    text("- C4（更多样化；回顾：基于规则的过滤）")
    text("- GitHub：保留宽松许可，基于手动规则过滤")
    text("- Wikipedia：2022 年 6-8 月，20 种语言，手动过滤")
    text("- Project Gutenberg 和 Books3（来自 The Pile）")
    text("- arXiv：删除评论、内联扩展宏、参考文献")
    text("- Stack Exchange：28 个最大的网站，按分数排序答案")
    text("结果：1.2T tokens")

    text("由 Together 的 RedPajama v1 复制 "), link("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    text("Cerebras 的 [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)：通过去重（MinHashLSH）得到的 RedPajama v1 的 6270 亿子集")

    text("无关：RedPajama v2 基于 84 个 CommonCrawl 快照有 30T tokens，最小过滤，大量质量信号 "), article_link("https://github.com/togethercomputer/RedPajama-Data")


def refinedweb():
    text("RefinedWeb "), link("https://arxiv.org/pdf/2306.01116") 
    text("- 要点：网络数据就是你所需要的一切")
    text("- [示例](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("- 使用 trafilatura 进行 HTML->文本，提取内容（WARC 而不是 WET 文件）")
    text("- 过滤：Gopher 规则，避免基于 ML 的过滤以避免偏见")
    text("- 使用 MinHash 对 5-grams 进行模糊去重")
    text("发布 6000 亿（共 5T）tokens")

    text("FineWeb "), article_link("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    text("- 最初作为 RefinedWeb 的复制，但改进了它")
    text("- 95 个 Common Crawl 转储")
    text("- URL 过滤，语言 ID（如果 p(en) > 0.65 则保留）")
    text("- 过滤：Gopher、C4、更多手动规则")
    text("- 通过 MinHash 进行模糊去重")
    text("- 匿名化电子邮件和公共 IP 地址（PII）")
    text("结果：15T tokens")


def dolma():
    text("Dolma "), link("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png", width=700)

    text("- Reddit：来自 Pushshift 项目（2005-2023），分别包含提交和评论")
    text("- PeS2o：来自 Semantic Scholar 的 4000 万篇学术论文")
    text("- C4、Project Gutenberg、Wikipedia/Wikibooks")

    text("Common Crawl 处理")
    text("- 语言识别（fastText 分类器），保留英语")
    text("- 质量过滤（Gopher、C4 规则），避免基于模型的过滤")
    text("- 使用规则和 Jigsaw 分类器进行毒性过滤")
    text("- 使用 Bloom filters 去重")

    text("结果：3T tokens")

def dclm():
    text("DataComp-LM "), link(dclm_2024)
    text("- 目标：定义一个标准数据集来尝试不同的数据处理算法")
    text("- 处理 CommonCrawl 以生成 DCLM-pool（240T tokens）")
    text("- DCLM-baseline：使用质量分类器过滤 DCLM-pool")
    image("images/dclm-filter.png", width=800)

    text("### 基于模型的过滤")
    text("正例（20 万）：")
    text("- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)：主要是 GPT-4 生成的指令数据（[示例](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train)）")
    text("- [ELI5](https://www.reddit.com/r/explainlikeimfive/)：带有好奇问题和答案的 subreddit（[示例](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train)）")
    text("负例（20 万）：")
    text("- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("结果：3.8T tokens")

    text("训练了一个 fastText 分类器，在所有 DCLM-pool 上运行")
    text("这个质量分类器优于其他过滤方法：")
    image("images/dclm-quality.png", width=600)


def nemotron_cc():
    text("Nemotron-CC "), link(nemotron_cc_2024)
    text("- FineWebEdu 和 DCLM 过滤过于激进（删除 90% 的数据）")
    text("- 需要更多 tokens（但保持质量）")
    text("- 对于 HTML -> 文本，使用 jusText（不是 trafilatura），因为它返回更多 tokens")

    text("分类器集成")
    text("- 提示 Nemotron-340B-instruct 根据教育价值对 FineWeb 文档进行评分，蒸馏到更快的模型")
    text("- DCLM 分类器")

    text("合成数据重述")
    text("- 对于低质量数据，使用 LM 重述低质量数据")
    text("- 对于高质量数据，使用 LM 生成任务（QA 对、提取关键信息等）")

    text("结果：6.3T tokens（HQ 子集为 1.1T）")
    text("作为参考，Llama 3 在 15T 上训练，Qwen3 在 36T 上训练")
    image("images/nemotron-results.png", width=800)


def copyright():
    text("围绕生成式 AI 有很多诉讼，主要是关于版权 "), article_link("https://www.bakerlaw.com/services/artificial-intelligence-ai/case-tracker-artificial-intelligence-copyrights-and-class-actions/")

    text("### 知识产权法")
    text("- 目标：*激励*知识产品的创造")
    text("- 知识产权类型：版权、专利、商标、商业秘密。")

    text("### 版权法")
    text("- 可以追溯到 1709 年的英格兰（安妮法令），首次由政府和法院监管 "), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    text("- 在美国，最近的：1976 年版权法 "), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    text("- 版权保护适用于'固定在任何有形表达媒介中的原创作品，现在已知或以后开发的，可以直接或借助机器或设备感知、复制或以其他方式传达'")

    text("- 原创作品，因此集合不受版权保护（例如，电话簿），除非在选择或排列上有一些创造性")
    text("- 版权适用于表达，而不是想法（例如，quicksort）")

    text("- 范围从'已发布'（1909）扩展到'固定'（1976）")
    text("- 版权保护不需要注册（与专利相反）")
    text("- 版权的门槛极低（例如，你的网站受版权保护）")

    text("- 创作者在起诉他人侵犯版权之前需要注册")
    text("- 注册费用为 $65 "), article_link("https://www.copyright.gov/about/fees.html")
    text("- 持续 75 年，然后版权到期，成为公共领域的一部分（莎士比亚、贝多芬的作品，Project Gutenberg 的大部分等）")

    text("总结：互联网上的大多数东西实际上都受版权保护。")

    text("如何使用受版权保护的作品：")
    text("1. 获得许可。")
    text("2. 诉诸合理使用条款。")

    text("## 许可证")
    text("- 许可证（来自合同法）由许可方授予被许可方。")
    text("- 实际上，'许可证是不起诉的承诺'。")

    text("- Creative Commons 许可证允许免费分发受版权保护的作品。")
    text("- 示例：Wikipedia、Open Courseware、Khan Academy、Free Music Archive、Flickr 的 3.07 亿张图片、MusicBrainz 的 3900 万张图片、YouTube 的 1000 万个视频等。")
    text("- 由 Lessig 和 Eldred 于 2001 年创建，以连接公共领域和现有版权")

    text("许多模型开发者为训练基础模型授权数据")
    text("- Google 和 Reddit "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- OpenAI 和 Shutterstock "), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    text("- OpenAI 和 StackExchange "), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    text("## 合理使用（第 107 条）")
    text("确定是否适用合理使用的四个因素：")
    text("1. 使用的目的和性质（教育优于商业，转化性优于复制性）")
    text("2. 受版权保护作品的性质（事实性优于虚构性，非创造性优于创造性）")
    text("3. 使用原作品部分的数量和实质性（使用片段优于使用整个作品）")
    text("4. 使用对原作品市场（或潜在市场）的影响")

    text("合理使用的示例：")
    text("- 你看了一部电影并写了一个摘要")
    text("- 重新实现一个算法（想法）而不是复制代码（表达）")
    text("- Google Books 索引并显示片段（Authors Guild v. Google 2002-2013）")

    text("版权不是关于逐字记忆")
    text("- 情节和角色（例如，哈利波特）可以受版权保护")
    text("- 戏仿可能是合理使用")
    text("版权是关于语义（和经济学）")

    text("基础模型的考虑因素：")
    text("- 复制数据（训练的第一步）已经是违规，即使你不对它做任何事情。")
    text("- 训练 ML 模型是转化性的（远不止是复制/粘贴）")
    text("- ML 系统对想法（例如，停车标志）感兴趣，而不是具体表达（例如，特定停车标志图像的确切艺术选择）。")
    text("问题：语言模型肯定会影响市场（作家、艺术家），无论版权如何")

    text("## 服务条款")
    text("- 即使你有许可证或可以诉诸作品的合理使用，服务条款也可能施加额外的限制。")
    text("- 示例：YouTube 的服务条款禁止下载视频，即使视频在 Creative Commons 下获得许可。")

    text("进一步阅读：")
    text("- [CS324 课程笔记](https://stanford-cs324.github.io/winter2022/lectures/legality/)")
    text("- Fair learning [[Lemley & Casey](https://texaslawreview.org/fair-learning/)]")
    text("- Foundation models and fair use "), link("https://arxiv.org/pdf/2303.15715")
    text("- The Files are in the Computer "), link("https://arxiv.org/abs/2404.12590")


def long_context():
    text("对长上下文的需求（想要对书籍进行问答）")
    text("- DeepSeek v3 有 128K tokens")
    text("- Claude 3.5 Sonnet 有 200K tokens")
    text("- Gemini 1.5 Pro 有 1.5M tokens")

    text("Transformers 与序列长度呈二次方关系")
    text("在长上下文上预训练效率不高，希望稍后添加长上下文")

    text("LongLoRA "), link("https://arxiv.org/pdf/2309.12307")
    text("- 将 Llama2 7B 的上下文长度从 4K 扩展到 100K tokens")
    text("- 使用移位稀疏注意力（图 2）、位置插值 [Chen+ 2023]")
    text("- 在长文档上训练：PG-19（书籍）和 Proof-Pile（数学）")


def tasks():
    text("TL;DR：将大量现有的 NLP 数据集转换为提示")

    text("Super-Natural Instructions "), link("https://arxiv.org/pdf/2204.07705")
    text("- 数据集：1.6K+ 任务（图 2）"), named_link("dataset", "https://huggingface.co/datasets/Muennighoff/natural-instructions")
    text("- 在 k-shot learning 上微调 T5（Tk-instruct）")
    text("- 社区贡献的任务（通过 GitHub）")
    text("- 每个任务的示例来自现有数据集并转换为模板化提示")
    text("- 尽管小得多，但优于 InstructGPT（?）")

    text("Flan 2022 "), link("https://arxiv.org/pdf/2301.13688")
    text("- 数据集：1.8K+ 任务 "), named_link("dataset", "https://huggingface.co/datasets/Muennighoff/flan")
    text("- 在数据集的 zero-shot、few-shot、chain-of-thought 版本上微调 T5（图 7）")


def instruction_chat():
    text("TL;DR：更开放的指令，大量使用合成数据")

    text("Alpaca "), link(alpaca)
    text("- 使用 self-instruct 从 text-davinci-003 获得的 52K 示例数据集 "), link("https://arxiv.org/pdf/2212.10560")
    text("- 在此数据集上微调 LLaMA 7B")

    text("Vicuna "), article_link("https://lmsys.org/blog/2023-03-30-vicuna/")
    text("- 在来自 [ShareGPT](https://sharegpt.com/) 的 70K 对话上微调 LLaMA（用户分享他们的 ChatGPT 对话；现已弃用）")

    text("Baize "), link("https://arxiv.org/pdf/2304.01196")
    text("- 使用 self-chat 从 GPT-3.5 生成数据集（111.5K 示例）（以 Quora 和 StackOverflow 问题为种子）")
    text("- 在此数据集上微调 LLaMA")

    text("WizardLM "), link("https://arxiv.org/pdf/2304.12244")
    text("- Evol-Instruct 数据集（'进化'问题以增加广度/难度）（图 1）")
    text("- 在此数据集上微调 LLaMA")

    text("MAmmoTH2 "), link("https://arxiv.org/pdf/2405.03548")
    text("- 策划 WebInstruct，来自 Common Crawl 的 1000 万条指令")
    text("- 过滤：在测验网站上训练 fastText 分类器")
    text("- 提取：使用 GPT-4 和 Mixtral 提取 QA 对")
    text("- 在此数据上微调 Mistral 7B")
    text("- 提升数学性能")

    text("OpenHermes 2.5")
    text("- 许多数据集的聚合 "), named_link("dataset", "https://huggingface.co/datasets/teknium/openhermes")
    text("- 在来自 GPT-4 的 100 万个示例上微调 Mistral 7B "), named_link("model", "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B")

    text("Llama 2 chat "), link("https://arxiv.org/pdf/2307.09288")
    text("- 来自基于供应商的标注的 27,540 个高质量指令数据示例")
    text("- 据说比使用来自开放数据集的数百万个示例更好")
    text("- 本可以标注更少的数据并为获取 RLHF 数据节省更多精力")

    text("Llama-Nemotron 后训练数据 [[NVIDIA, 2024](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)]")
    text("- 提示：公共数据集（例如，WildChat）或合成生成，然后过滤")
    text("- 从 Llama、Mixtral、DeepSeek r1、Qwen 生成合成响应（商业上可行，不像 GPT-4）")
    text("- 包含推理轨迹")
    text("- [示例](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/viewer/SFT/code)")


if __name__ == "__main__":
    main()
