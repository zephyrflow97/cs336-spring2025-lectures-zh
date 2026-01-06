# Spring 2025 CS336 lectures

本仓库包含 "Stanford CS336: Language modeling from scratch" 的课程材料。

## 非可执行课程（ppt/pdf）

位于 `nonexecutable/` 目录下的 PDF 文件

## 可执行课程

位于根目录下的 `lecture_*.py` 文件

### 快速开始

#### 1. 环境设置

首先，确保安装所有依赖：

```bash
# 激活虚拟环境（如果有）
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**注意**：如果你使用 macOS ARM64（Apple Silicon），`triton` 库不可用。这是正常的，相关功能会被自动跳过。

#### 2. 生成课程内容

编译一个课程：

```bash
python execute.py -m lecture_01
```

这会生成 `var/traces/lecture_01.json` 并缓存相应的图片。

#### 3. 查看课程内容

有两种方式查看生成的课程：

**方式 A：使用本地开发服务器（推荐）**

```bash
cd trace-viewer
npm run dev
```

然后在浏览器中访问：
- 主页：`http://localhost:5173/`
- 查看特定课程：`http://localhost:5173/?trace=var/traces/lecture_01.json`

**方式 B：直接打开静态页面**

在浏览器中打开 `index.html` 文件，但这种方式可能有跨域限制。

### 在集群上运行

如果你想在集群上运行：

```bash
./remote_execute.sh lecture_01
```

这会将文件复制到 slurm 集群，在那里运行，然后将结果复制回来。你需要设置适当的环境并调整一些配置才能使其工作（这些说明不完整）。

### 前端开发

如果你需要修改 JavaScript 代码：

**首次安装**：

```bash
npm create vite@latest trace-viewer -- --template react
cd trace-viewer
npm install
```

**本地开发**：

```bash
cd trace-viewer
npm run dev
```

然后访问 `http://localhost:5173/` 查看更改。

**部署到主网站**：

```bash
cd trace-viewer
npm run build
git add dist/assets
# 然后提交到仓库，它应该会显示在网站上
```

## 可用课程列表

- `lecture_01.py` - 课程 1
- `lecture_02.py` - 课程 2
- `lecture_06.py` - 课程 6：基准测试/性能分析 + 编写 kernel
- `lecture_08.py` - 课程 8
- `lecture_10.py` - 课程 10：推理优化
- `lecture_12.py` - 课程 12
- `lecture_13.py` - 课程 13
- `lecture_14.py` - 课程 14
- `lecture_17.py` - 课程 17

## 常见问题

### Q: 为什么 triton 无法安装？
A: Triton 目前不支持 macOS ARM64（Apple Silicon）。代码已经过修改，会自动跳过 Triton 相关功能。你仍然可以学习课程的其他部分。

### Q: 如何查看已生成的课程？
A: 运行 `cd trace-viewer && npm run dev`，然后在浏览器中访问 `http://localhost:5173/`。

### Q: 生成的 JSON 文件在哪里？
A: 在 `var/traces/` 目录下，例如 `var/traces/lecture_01.json`。
