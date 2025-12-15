# 本地 AI 智能文献与图像管理助手 (Local Multimodal AI Agent)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于多模态神经网络的本地智能助手，用于语义搜索和自动分类管理文献与图像。

## 🌟 核心功能

### 📚 智能文献管理
- **语义搜索**: 使用自然语言查询相关论文（如 "Transformer 的核心架构"）
- **自动分类**: 根据内容自动将论文归类到指定主题文件夹
- **批量整理**: 一键整理混乱的文献库
- **向量数据库**: 持久化存储，支持高效检索

### 🖼️ 智能图像管理
- **以文搜图**: 通过自然语言描述查找图片（如 "海边的日落"）
- **多模态匹配**: 基于 CLIP 的图文语义匹配

## 🚀 快速开始

### 前置要求
- Python 3.9 或更高版本
- [UV](https://github.com/astral-sh/uv) 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone <your-repo-url>
cd multimodal-ai-agent
```

2. **安装依赖**
```bash
uv sync
```

这将自动安装所有必需的依赖包，包括：
- `sentence-transformers`: 文本嵌入模型
- `chromadb`: 向量数据库
- `pypdf`: PDF 文本提取
- `torch` & `torchvision`: 深度学习框架
- `click`: CLI 工具

### 基本使用

#### 1. 添加并分类单篇论文

```bash
uv run python main.py add_paper path/to/paper.pdf --topics "CV,NLP,RL"
```

参数说明：
- `path/to/paper.pdf`: PDF 文件路径
- `--topics`: 逗号分隔的主题列表
- `--copy`: 复制而非移动原文件

**示例：**
```bash
# 添加一篇关于计算机视觉的论文
uv run python main.py add_paper papers/attention_is_all_you_need.pdf --topics "CV,NLP,RL"

# 保留原文件
uv run python main.py add_paper papers/resnet.pdf --topics "CV,NLP,RL" --copy
```

#### 2. 批量整理文件夹

```bash
uv run python main.py organize_papers folder_path --topics "CV,NLP,RL"
```

**示例：**
```bash
# 整理 Downloads 文件夹中的所有 PDF
uv run python main.py organize_papers ~/Downloads --topics "CV,NLP,RL"
```

#### 3. 语义搜索论文

```bash
uv run python main.py search_paper "transformer architecture"
```

可选参数：
- `--top-k`: 返回结果数量（默认 5）

**示例：**
```bash
# 搜索关于 Transformer 的论文
uv run python main.py search_paper "what is the core architecture of Transformer"

# 返回前 10 个结果
uv run python main.py search_paper "deep learning optimization" --top-k 10
```

#### 4. 以文搜图

首先，将图片放入 `storage/images` 目录：

```bash
# 创建图片目录（如果不存在）
mkdir -p storage/images

# 复制图片到目录
cp ~/Pictures/*.jpg storage/images/
```

然后搜索：

```bash
uv run python main.py search_image "sunset by the sea"
```

**示例：**
```bash
# 搜索海边日落的图片
uv run python main.py search_image "sunset by the sea"

# 搜索猫的图片
uv run python main.py search_image "cute cat" --top-k 10
```

## 📁 项目结构

```
multimodal-ai-agent/
├── main.py                 # CLI 主入口
├── pyproject.toml          # 项目配置和依赖
├── README.md               # 项目文档
├── app/
│   ├── __init__.py
│   ├── config.py           # 配置设置
│   ├── embeddings.py       # 文本和图像嵌入
│   ├── chroma_store.py     # ChromaDB 向量存储
│   └── utils.py            # 工具函数
├── storage/
│   ├── papers/             # 论文存储（按主题分类）
│   │   ├── CV/
│   │   ├── NLP/
│   │   └── RL/
│   └── images/             # 图片存储
└── data/
    └── chroma/             # ChromaDB 数据目录
```

## 🛠️ 技术栈

### 核心模型
- **文本嵌入**: `sentence-transformers/all-MiniLM-L6-v2`
  - 轻量级，快速，适合 CPU
  - 384 维向量空间
  
- **图像嵌入**: `openai/clip-vit-base-patch32`
  - CLIP 模型，支持图文匹配
  - 512 维向量空间

### 数据库
- **ChromaDB**: 嵌入式向量数据库
  - 无需服务器部署
  - 自动持久化
  - 支持余弦相似度搜索

## ⚙️ 配置选项

### 环境变量

你可以通过环境变量自定义模型：

```bash
# 文本嵌入模型
export TEXT_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 图像嵌入模型
export IMAGE_MODEL_NAME="openai/clip-vit-large-patch14"

# 设备选择 (cpu, cuda, mps)
export DEVICE="cuda"

# 运行
uv run python main.py search_paper "your query"
```

### 配置文件

编辑 `app/config.py` 以修改默认设置：

```python
# 文本分块设置
CHUNK_SIZE = 1000        # 每个文本块的字符数
CHUNK_OVERLAP = 200      # 文本块之间的重叠

# 模型设置
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cpu"
```

## 📊 使用示例

### 示例 1: 构建个人论文库

```bash
# 1. 整理现有论文
uv run python main.py organize_papers ~/Documents/Papers --topics "CV,NLP,RL,Theory"

# 2. 搜索相关论文
uv run python main.py search_paper "attention mechanism in neural networks"

# 3. 添加新论文
uv run python main.py add_paper new_paper.pdf --topics "CV,NLP,RL,Theory"
```

### 示例 2: 图片库管理

```bash
# 1. 准备图片
mkdir -p storage/images
cp ~/Pictures/*.jpg storage/images/

# 2. 搜索特定场景
uv run python main.py search_image "mountain landscape"

# 3. 搜索特定物体
uv run python main.py search_image "red sports car"
```

## 🔧 进阶功能

### GPU 加速

如果你有 NVIDIA GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 设置使用 GPU
export DEVICE="cuda"
```

### 更强大的模型

替换为更大的模型以获得更好的效果：

```python
# 在 app/config.py 中
TEXT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 更准确
IMAGE_MODEL_NAME = "openai/clip-vit-large-patch14"           # 更大的 CLIP
```

## 🐛 故障排除

### 问题 1: 无法提取 PDF 文本

**原因**: PDF 可能是扫描版或加密的

**解决方案**:
- 使用 OCR 工具预处理
- 确保 PDF 不是图片格式

### 问题 2: 内存不足

**解决方案**:
```python
# 减小批处理大小
# 在 app/embeddings.py 中调整 batch_size
embeddings = self.model.encode(texts, batch_size=8)  # 默认是 32
```

### 问题 3: 模型下载慢

**解决方案**:
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

## 📝 开发建议

### 添加新功能

1. **添加新的主题分类**
   - 在调用时指定新主题即可

2. **自定义分类逻辑**
   - 修改 `main.py` 中的 `classify_paper` 函数

3. **添加 OCR 支持**
   ```python
   # 在 app/utils.py 中添加
   from PIL import Image
   import pytesseract
   
   def extract_image_text(image_path):
       return pytesseract.image_to_string(Image.open(image_path))
   ```

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题，请提交 Issue 或联系维护者。

---

**祝使用愉快！🎉**