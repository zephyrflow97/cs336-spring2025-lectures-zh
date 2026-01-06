import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def get_device(index: int = 0) -> torch.device:
    """如果可能，尝试使用 GPU，否则使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

class MLP(nn.Module):
    """简单的 MLP：linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        # 标记整个前向传播
        for i, layer in enumerate(self.layers):
            # 分别标记每一层的计算
            with nvtx.range(f"layer_{i}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int, use_optimizer: bool = False):
    """在 MLP 上运行前向和反向传播。
    
    参数：
        dim: 每层的维度
        num_layers: linear+GeLU 层的数量
        batch_size: 一次处理的样本数量
        num_steps: 前向/反向迭代次数
        use_optimizer: 是否使用 Adam 优化器更新权重
    """
    # 定义一个模型（随机权重）
    with nvtx.range("define_model"):
        model = MLP(dim, num_layers).to(get_device())
    
    # 如果需要，初始化优化器
    optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None

    # 定义一个输入（随机）
    with nvtx.range("define_input"):
        x = torch.randn(batch_size, dim, device=get_device())

    # 运行模型 `num_steps` 次
    for step in range(num_steps):
        if step > 10:
            # 在 10 次预热迭代后开始性能分析
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"step_{step}")
        
        # 梯度清零
        if use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)

        # 前向传播
        with nvtx.range("forward"):
            y = model(x).mean()

        # 反向传播
        with nvtx.range("backward"):
            y.backward()

        # 如果启用，执行优化器步骤
        if use_optimizer:
            with nvtx.range("optimizer_step"):
                #print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
        
        nvtx.range_pop()

def main():
    # 如果 GPU 可用，运行更大的模型
    if torch.cuda.is_available():
        print("在 GPU 上运行")
        run_mlp(dim=4096, num_layers=64, batch_size=1024, num_steps=15, use_optimizer=True)
    else:
        print("在 CPU 上运行")
        run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=15, use_optimizer=True)

if __name__ == "__main__":
    main()