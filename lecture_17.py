import os
import sys
from typing import Callable
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import softmax
from einops import einsum, rearrange, repeat
from execute_util import text, link, image
from lecture_util import named_link
from references import ppo2017, grpo, qwen3, llama3
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    text("上节课：从可验证奖励的强化学习概述 (policy gradient)")
    text("本节课：深入探讨 policy gradient 的机制（例如 GRPO）")

    rl_setup_for_language_models()
    policy_gradient()
    training_walkthrough()

    text("总结")
    text("- Reinforcement learning 是超越人类能力的关键")
    text("- **如果**你能衡量它，你就能优化它")
    text("- Policy gradient 框架在概念上很清晰，只需要 baselines 来减少方差")
    text("- RL 系统比预训练复杂得多（推理工作负载，管理多个模型）")

    text("最后两节课：")
    text("- Junyang Lin (Qwen) "), link(qwen3)
    text("- Mike Lewis (Llama) "), link(llama3)


def rl_setup_for_language_models():
    text("**状态 (State)** s: prompt + 目前为止生成的响应")
    text("**动作 (Action)** a: 生成下一个 token")

    text("**奖励 (Rewards)** R: 响应的好坏程度；我们将关注：")
    text("- Outcome rewards（结果奖励），取决于整个响应")
    text("- Verifiable rewards（可验证奖励），其计算是确定性的")
    text("- Discounting 和 bootstrapping 的概念在这里不太适用")
    text("示例：\"... 因此，答案是 3 英里。\"")

    text("**转移概率 (Transition probabilities)** T(s' | s, a): 确定性的 s' = s + a")
    text("- 可以进行规划 / 测试时计算（与机器人不同）")
    text("- 状态实际上是虚构的（与机器人不同），所以有很大的灵活性")

    text("**策略 (Policy)** π(a | s): 就是一个语言模型（经过微调）")

    text("**Rollout/episode/trajectory（轨迹）**: s → a → ... → a → a → R")
    text("**目标 (Objective)**: 最大化期望奖励 E[R]")
    text("（其中期望是对 prompts s 和响应 tokens a 求得的）")


def policy_gradient():
    text("为了符号简洁，让 *a* 表示整个响应。")

    text("我们想要相对于策略 π 最大化期望奖励：")
    text("E[R] = ∫ p(s) π(a | s) R(s, a)")

    text("显而易见的做法是求梯度：")
    text("∇ E[R] = ∫ p(s) ∇ π(a | s) R(s, a)")
    text("∇ E[R] = ∫ p(s) π(a | s) ∇ log π(a | s) R(s, a)")
    text("∇ E[R] = E[∇ log π(a | s) R(s, a)]")

    text("朴素的 policy gradient：")
    text("- 采样 prompt s，采样响应 a ~ π(a | s)")
    text("- 基于 ∇ log π(a | s) R(s, a) 更新参数（与 SFT 相同，但由 R(s, a) 加权）")

    text("设定：R(s, a) ∈ {0, 1} = 响应是否正确")
    text("- 朴素的 policy gradient 只在正确响应上更新")
    text("- 类似 SFT，但数据集随着策略变化而变化")

    text("挑战：高噪声/方差")
    text("在这种设定下，奖励稀疏（少数响应获得奖励 1，大多数获得 0）")
    text("相比之下：在 RLHF 中，奖励模型（从成对偏好中学习）更加连续")

    text("### Baselines（基线）")
    text("回顾 ∇ E[R] = E[∇ log π(a | s) R(s, a)]")
    text("∇ log π(a | s) R(s, a) 是 ∇ E[R] 的无偏估计，但也许有其他方差更低的估计...")

    text("示例：两个状态")
    text("- s1: a1 → 奖励 11, a2 → 奖励 9")
    text("- s2: a1 → 奖励 0, a2 → 奖励 2")
    text("不希望 s1 → a2（奖励 9），因为 a1 更好，希望 s2 → a2（奖励 2），但 9 > 2")

    text("想法：最大化基线化的奖励：E[R - b(s)]")
    text("这只是 E[R] 平移了一个不依赖于策略 π 的常数 E[b(s)]")
    text("我们基于 ∇ log π(a | s) (R(s, a) - b(s)) 进行更新")

    text("我们应该使用什么 b(s)？")

    text("示例：两个状态")
    text("假设 (s, a) 上的均匀分布且 |∇ π(a | s)| = 1")
    naive_variance = torch.std(torch.tensor([11., 9, 0, 2]))  # @inspect naive_variance
    text("定义 baseline b(s1) = 10, b(s2) = 1")
    baseline_variance = torch.std(torch.tensor([11. - 10, 9 - 10, 0 - 1, 2 - 1]))  # @inspect baseline_variance
    text(f"方差从 {naive_variance:.3f} 降低到 {baseline_variance:.3f}")

    text("最优 b*(s) = E[(∇ π(a | s))^2 R | s] / E[(∇ π(a | s))^2 | s]（对于单参数模型）")
    text("这很难计算...")
    text("...所以启发式方法是使用平均奖励：")
    text("b(s) = E[R | s]")
    text("这仍然很难计算，必须进行估计。")

    text("### Advantage functions（优势函数）")
    text("这种 b(s) 的选择与 advantage functions 有关。")
    text("- V(s) = E[R | s] = 从状态 s 的期望奖励")
    text("- Q(s, a) = E[R | s, a] = 从状态 s 采取动作 a 的期望奖励")
    text("（注意：这里 Q 和 R 是相同的，因为我们假设 *a* 包含所有动作，并且我们有 outcome rewards。）")

    text("定义（advantage）：A(s, a) = Q(s, a) - V(s)")
    text("直觉：动作 a 比从状态 s 的期望好多少")

    text("如果 b(s) = E[R | s]，那么基线化的奖励与 advantage 相同！")
    text("E[R - b(s)] = A(s, a)")

    text("一般来说：")
    text("- 理想：E[∇ log π(a | s) R(s, a)]")
    text("- 估计：∇ log π(a | s) δ")
    text("δ 有多种选择，我们稍后会看到。")

    named_link("CS224R lecture notes", "https://cs224r.stanford.edu/slides/03_cs224r_policy_gradients_2025.pdf")


def training_walkthrough():
    text("Group Relative Policy Optimization (GRPO) "), link(grpo)
    text("- 对 PPO 的简化，移除了 critic（value function）")
    text("- 利用 LM 设置中的组结构（每个 prompt 有多个响应），这提供了一个自然的 baseline b(s)。")
    image("images/grpo-algorithm.png", width=700)

    simple_task()        # 定义一个简单任务
    simple_model()       # 定义一个简单模型

    text("现在让我们定义 GRPO 算法。")
    run_policy_gradient(num_epochs=1, num_steps_per_epoch=1)

    text("让我们实际训练一些模型。")
    experiments()


def simple_task():
    text("任务：对 n 个数字排序")

    text("Prompt：n 个数字")
    prompt = [1, 0, 2]
    text("Response：n 个数字")
    response = [0, 1, 2]

    text("奖励应该捕捉响应与排序结果的接近程度。")

    text("定义一个奖励函数，返回响应与真实值匹配的位置数。")
    reward = sort_distance_reward([3, 1, 0, 2], [0, 1, 2, 3])  # @inspect reward
    reward = sort_distance_reward([3, 1, 0, 2], [7, 2, 2, 5])  # @inspect reward  @stepover
    reward = sort_distance_reward([3, 1, 0, 2], [0, 3, 1, 2])  # @inspect reward  @stepover

    text("定义一个给予更多部分分的替代奖励函数。")
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [0, 1, 2, 3])  # @inspect reward
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [7, 2, 2, 5])  # @inspect reward  @stepover
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [0, 3, 1, 2])  # @inspect reward  @stepover

    text("注意，第二个奖励函数对第 3 个响应给予的分数比第一个奖励函数更多。")


def simple_model():
    text("定义一个将 prompts 映射到 responses 的简单模型")
    text("- 假设固定的 prompt 和 response 长度")
    text("- 使用每个位置独立的参数捕捉位置信息")
    text("- 独立解码响应中的每个位置（非自回归）")

    model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)

    text("从一个 prompt s 开始")
    prompts = torch.tensor([[1, 0, 2]])  # [batch pos]

    text("生成响应 a")
    torch.manual_seed(10)
    responses = generate_responses(prompts=prompts, model=model, num_responses=5)  # [batch trial pos]  @inspect responses

    text("计算这些响应的奖励 R：")
    rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=sort_inclusion_ordering_reward)  # [batch trial]  @inspect rewards

    text("根据奖励 R 计算 deltas δ（用于执行更新）")
    deltas = compute_deltas(rewards=rewards, mode="rewards")  # [batch trial]  @inspect deltas
    deltas = compute_deltas(rewards=rewards, mode="centered_rewards")  # [batch trial]  @inspect deltas
    deltas = compute_deltas(rewards=rewards, mode="normalized_rewards")  # [batch trial]  @inspect deltas
    deltas = compute_deltas(rewards=rewards, mode="max_rewards")  # [batch trial]  @inspect deltas

    text("计算这些响应的对数概率：")
    log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]  @inspect log_probs

    text("计算损失以便用于更新模型参数")
    loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="naive")  # @inspect loss

    freezing_parameters()

    old_model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)  # 假装这是一个旧的 checkpoint @stepover
    old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=old_model)  # @stepover
    loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="unclipped", old_log_probs=old_log_probs)  # @inspect loss
    loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="clipped", old_log_probs=old_log_probs)  # @inspect loss

    text("有时，我们可以使用显式的 KL penalty 来正则化模型。")
    text("如果你想通过 RL 向模型中添加新能力，但又不希望它忘记原有能力，这会很有用。")
    text("KL(p || q) = E_{x ~ p}[log(p(x)/q(x))]")
    text("KL(p || q) = E_{x ~ p}[-log(q(x)/p(x))]")
    text("KL(p || q) = E_{x ~ p}[q(x)/p(x) - log(q(x)/p(x)) - 1] 因为 E_{x ~ p}[q(x)/p(x)] = 1")
    kl_penalty = compute_kl_penalty(log_probs=log_probs, ref_log_probs=old_log_probs)  # @inspect kl_penalty

    text("总结：")
    text("- 生成响应")
    text("- 计算奖励 R 和 δ（rewards, centered rewards, normalized rewards, max rewards）")
    text("- 计算响应的 log probs")
    text("- 从 log probs 和 δ 计算损失（naive, unclipped, clipped）")


def freezing_parameters():
    text("动机：在 GRPO 中你会看到比率：p(a | s) / p_old(a | s)")
    text("在优化时，重要的是冻结 p_old 并且不对其求导")
    w = torch.tensor(2., requires_grad=True)
    p = torch.nn.Sigmoid()(w)
    p_old = torch.nn.Sigmoid()(w)
    ratio = p / p_old
    ratio.backward()
    grad = w.grad  # @inspect grad

    text("正确的做法：")
    w = torch.tensor(2., requires_grad=True)
    p = torch.nn.Sigmoid()(w)
    with torch.no_grad():  # 重要：将 p_old 视为常数！
        p_old = torch.nn.Sigmoid()(w)
    ratio = p / p_old
    ratio.backward()
    grad = w.grad  # @inspect grad


def compute_reward(prompts: torch.Tensor, responses: torch.Tensor, reward_fn: Callable[[list[int], list[int]], float]) -> torch.Tensor:
    """
    计算奖励
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        rewards (float[batch trial])
    """
    batch_size, num_responses, _ = responses.shape
    rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(num_responses):
            rewards[i, j] = reward_fn(prompts[i, :], responses[i, j, :])
    return rewards


def sort_distance_reward(prompt: list[int], response: list[int]) -> float:  # @inspect prompt, @inspect response
    """
    返回响应与 ground_truth = sorted(prompt) 的接近程度。
    具体来说，计算响应与真实值匹配的位置数。
    """
    assert len(prompt) == len(response)
    ground_truth = sorted(prompt)
    return sum(1 for x, y in zip(response, ground_truth) if x == y)


def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:  # @inspect prompt, @inspect response
    """
    返回响应与 ground_truth = sorted(prompt) 的接近程度。
    """
    assert len(prompt) == len(response)

    # 对于 prompt 中出现在 response 中的每个 token 给一分
    inclusion_reward = sum(1 for x in prompt if x in response)  # @inspect inclusion_reward

    # 对于 response 中已排序的每对相邻元素给一分
    ordering_reward = sum(1 for x, y in zip(response, response[1:]) if x <= y)  # @inspect ordering_reward

    return inclusion_reward + ordering_reward


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, prompt_length: int, response_length: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 对于每个位置，我们有一个用于编码的矩阵和一个用于解码的矩阵
        self.encode_weights = nn.Parameter(torch.randn(prompt_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))
        self.decode_weights = nn.Parameter(torch.randn(response_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: int[batch pos]
        Returns:
            logits: float[batch pos vocab]
        """
        # 嵌入 prompts
        embeddings = self.embedding(prompts)   # [batch pos dim]

        # 使用每个 prompt 位置的矩阵进行变换，折叠成一个向量
        encoded = einsum(embeddings, self.encode_weights, "batch pos dim1, pos dim1 dim2 -> batch dim2")

        # 转换为每个响应位置的一个向量
        decoded = einsum(encoded, self.decode_weights, "batch dim2, pos dim2 dim1 -> batch pos dim1")

        # 转换为 logits（输入和输出共享 embeddings）
        logits = einsum(decoded, self.embedding.weight, "batch pos dim1, vocab dim1 -> batch pos vocab")

        return logits


def generate_responses(prompts: torch.Tensor, model: Model, num_responses: int) -> torch.Tensor:
    """
    生成响应
    Args:
        prompts (int[batch pos])
    Returns:
        generated responses: int[batch trial pos]

    示例 (batch_size = 3, prompt_length = 3, num_responses = 2, response_length = 4)
    p1 p1 p1 r1 r1 r1 r1
             r2 r2 r2 r2
    p2 p2 p2 r3 r3 r3 r3
             r4 r4 r4 r4
    p3 p3 p3 r5 r5 r5 r5
             r6 r6 r6 r6
    """
    logits = model(prompts)  # [batch pos vocab]
    batch_size = prompts.shape[0]

    # 对每个 [batch pos] 采样 num_responses（独立采样）
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    flattened_responses = torch.multinomial(softmax(flattened_logits, dim=-1), num_samples=num_responses, replacement=True)  # [batch pos trial]
    responses = rearrange(flattened_responses, "(batch pos) trial -> batch trial pos", batch=batch_size)
    return responses


def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, model: Model) -> torch.Tensor:
    """
    计算对数概率
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        log_probs (float[batch trial pos]) 在模型下的对数概率
    """
    # 在模型下计算响应的对数概率
    logits = model(prompts)  # [batch pos vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch pos vocab]

    # 复制以与响应对齐
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", trial=num_responses)  # [batch trial pos vocab]

    # 使用响应索引到 log_probs
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)  # [batch trial pos]

    return log_probs


def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:  # @inspect rewards
    """
    计算 deltas
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) 用于更新的类似 advantage 的量
    """
    if mode == "rewards":
        return rewards

    if mode == "centered_rewards":
        # 对每个 prompt (batch) 计算所有响应 (trial) 的均值
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        return centered_rewards

    if mode == "normalized_rewards":
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        std_rewards = rewards.std(dim=-1, keepdim=True)  # @inspect std_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)  # @inspect normalized_rewards
        return normalized_rewards

    if mode == "max_rewards":
        # 将每个 batch 中不是最大值的奖励清零
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards

    raise ValueError(f"Unknown mode: {mode}")


def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "unclipped":
        ratios = log_probs / old_log_probs  # [batch trial]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  # [batch trial]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")

        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()

    raise ValueError(f"Unknown mode: {mode}")

def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    计算 KL(model | ref_model) 的估计，其中模型由以下给出：
        log_probs [batch trial pos vocab]
        ref_log_probs [batch trial pos vocab]
    使用估计：
        KL(p || q) = E_p[q/p - log(q/p) - 1]
    """
    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()


def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        compute_ref_model_period: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "rewards",
                        loss_mode: str = "naive",
                        kl_penalty: float = 0.0,
                        reward_fn: Callable[[list[int], list[int]], float] = sort_inclusion_ordering_reward,
                        use_cache: bool = False) -> tuple[str, str]:
    """使用 policy gradient 训练模型。
    返回：
    - 学习曲线图像的路径
    - 日志文件的路径
    """
    torch.manual_seed(5)

    image_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.png"
    log_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.txt"

    # 已经运行过，直接使用缓存
    if use_cache and os.path.exists(image_path) and os.path.exists(log_path):
        return image_path, log_path

    # 定义数据
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]])
    vocab_size = prompts.max() + 1
    prompt_length = response_length = prompts.shape[1]

    model = Model(vocab_size=vocab_size, embedding_dim=10, prompt_length=prompt_length, response_length=response_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    records = []
    ref_log_probs = None
    ref_model = None
    old_log_probs = None

    if use_cache:
        out = open(log_path, "w")
    else:
        out = sys.stdout

    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # 如果使用 KL penalty，需要获取参考模型（每隔几个 epoch 冻结一次）
        if kl_penalty != 0:
            if epoch % compute_ref_model_period == 0:
                ref_model = model.clone()

        # 采样响应并评估它们的奖励
        responses = generate_responses(prompts=prompts, model=model, num_responses=num_responses)  # [batch trial pos]
        rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=reward_fn)  # [batch trial]
        deltas = compute_deltas(rewards=rewards, mode=deltas_mode)  # [batch trial]

        if kl_penalty != 0:  # 在参考模型下计算
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=ref_model)  # [batch trial]

        if loss_mode != "naive":  # 在当前模型下计算（但在执行内部步骤时冻结）
            with torch.no_grad():
                old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]

        # 根据响应执行多个步骤
        for step in range(num_steps_per_epoch):
            log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
            loss = compute_loss(log_probs=log_probs, deltas=deltas, mode=loss_mode, old_log_probs=old_log_probs)  # @inspect loss
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs=log_probs, ref_log_probs=ref_log_probs)

            # 打印信息
            print_information(epoch=epoch, step=step, loss=loss, prompts=prompts, rewards=rewards, responses=responses, log_probs=log_probs, deltas=deltas, out=out)
            global_step = epoch * num_steps_per_epoch + step
            records.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "mean_reward": rewards.mean().item()})

            # 反向传播并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if use_cache:
        out.close()

    if use_cache:
        # 在两个子图中绘制步骤与损失和奖励的关系
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        rewards = [r["mean_reward"] for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失子图
        ax1.plot(steps, losses)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Train Loss")

        # 奖励子图
        ax2.plot(steps, rewards)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Mean Reward")
        ax2.set_title("Mean Reward")

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

    return image_path, log_path


def print_information(epoch: int, step: int, loss: torch.Tensor, prompts: torch.Tensor, rewards: torch.Tensor, responses: torch.Tensor, log_probs: torch.Tensor, deltas: torch.Tensor, out):
    print(f"epoch = {epoch}, step = {step}, loss = {loss:.3f}, reward = {rewards.mean():.3f}", file=out)
    if epoch % 1 == 0 and step % 5 == 0:
        for batch in range(prompts.shape[0]):
            print(f"  prompt = {prompts[batch, :]}", file=out)
            for trial in range(responses.shape[1]):
                print(f"    response = {responses[batch, trial, :]}, log_probs = {tstr(log_probs[batch, trial])}, reward = {rewards[batch, trial]}, delta = {deltas[batch, trial]:.3f}", file=out)


def tstr(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x[i]:.3f}" for i in range(x.shape[0])) + "]"


def experiments():
    text("让我们从基于原始奖励的更新开始。")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10, deltas_mode="rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)  # @stepover
    image(image_path, width=600), link(log_path)
    text("查看输出，你会发现到最后，我们并没有很好地学会排序（而且这仍然是训练集）。")

    text("让我们尝试使用 centered rewards。")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10, deltas_mode="centered_rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)  # @stepover
    image(image_path, width=600), link(log_path)
    text("这似乎有帮助，因为：")
    text("- 次优奖励会得到负梯度更新，并且")
    text("- 如果给定 prompt 的所有响应具有相同的奖励，那么我们不会更新。")
    text("总的来说，这更好，但我们仍然会陷入局部最优。")

    text("最后，让我们尝试按标准差归一化。")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10, deltas_mode="normalized_rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)  # @stepover
    image(image_path, width=600), link(log_path)
    text("这里没有太大区别，事实上，像 Dr. GRPO 这样的变体不执行此归一化以避免长度偏差（这里不是问题，因为所有响应的长度相同。"), link("https://arxiv.org/abs/2503.20783")

    text("总的来说，正如你所看到的，reinforcement learning 并非易事，你很容易陷入次优状态。")
    text("超参数可能可以调得更好...")


if __name__ == "__main__":
    main()
