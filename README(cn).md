# 信息检索系统项目

这是一个基于深度学习的信息检索系统项目，包含多种检索方法的实现与比较，包括传统的BM25算法、微调的Bi-Encoder模型、LLM重排以及三阶段混合检索架构。

## 项目概述

本项目旨在通过多种方法改进医疗领域的信息检索效果。我们实现了从传统的信息检索算法到最新的深度学习模型，再到大语言模型(LLM)辅助的检索技术，并对这些方法进行了全面的评估和比较。

## 项目结构

### 文件夹结构

- `data/` - 存放原始数据文件
- `datasets/` - 存放处理后的数据集
- `models/` - 存放训练好的模型文件
  - `finetuned-medical-retriever` - 微调后的医疗领域检索模型
- `results/` - 存放实验结果和评估指标

### Python脚本功能说明

- `load_data.py` - 加载NFCorpus医疗检索数据集
- `download_datasets.py` - 下载多个BEIR数据集
- `bm25_baseline.py` - 实现BM25基准模型
- `dense_retrieval_pretrained.py` - 使用预训练模型进行密集检索
- `dense_retrieval_finetuned.py` - 使用微调后模型进行密集检索
- `prepare_training_data.py` - 准备训练数据
- `train_biencoder_offline.py` - 训练Bi-Encoder模型
- `llm_reranking_improved.py` - 使用大语言模型进行重排
- `hybrid_retrieval.py` - 三阶段混合检索架构（BM25粗排 + Bi-Encoder精排 + LLM重排）
- `multi_dataset_evaluation.py` - 在多个数据集上评估模型性能
- `visualize_results.py` - 可视化结果对比

## 安装方法

首先，克隆此仓库：
```bash
git clone https://github.com/zhijiancui/IR-project.git
cd IR-project
```

然后安装所需的依赖项：
```bash
pip install -r requirements.txt
```

## 数据和模型文件

**重要说明：** 由于GitHub的文件大小限制，数据和模型文件未包含在仓库中。您需要单独下载或生成它们：

### 数据获取
本项目使用的数据集可通过我们的共享网盘获取：
- 链接：[[IR project](https://bhpan.buaa.edu.cn/link/AAD502EC85886C45818CF89E6957360695)]，提取码：nvbh
- 内容：
  - BEIR基准数据集`datasets.zip`
  - Bi-Encoder模型的训练数据`data.zip`
  - 结果数据和统计图表`results.zip`

下载后，请解压文件并放置在以下文件夹中：
```
IR project/
├── datasets/
├── data/
└── results/
```

或者，您可以使用 `download_datasets.py` 下载部分公共数据集。

### 模型获取
训练好的模型可通过以下方式获取：
- 链接：[[IR project](https://bhpan.buaa.edu.cn/link/AAD502EC85886C45818CF89E6957360695)]，提取码：nvbh
- 内容：
  - 微调后的医疗检索模型（`models/finetuned-medical-retriever/`）
  - 不同训练阶段的检查点文件
  - 预训练基础模型

下载后，请将模型放置在以下文件夹中：
```
models/
└── finetuned-medical-retriever/
    ├── 1_Pooling/
    ├── checkpoints/
    └── [模型文件]
```

### 模型训练
如果您希望自己训练模型：
1. 使用 `prepare_training_data.py` 准备训练数据
2. 使用 `train_biencoder_offline.py` 训练Bi-Encoder模型
3. 训练好的模型将保存到 `models/finetuned-medical-retriever/`

## 使用方法

### 环境准备
```bash
pip install -r requirements.txt
```

### 运行流程
1. 下载数据集：运行 `download_datasets.py`
2. 运行BM25基准：运行 `bm25_baseline.py`
3. 运行微调模型：运行 `dense_retrieval_finetuned.py`
4. 运行LLM重排：运行 `llm_reranking_improved.py`
5. 运行混合架构：运行 `hybrid_retrieval.py`
6. 查看可视化结果：运行 `visualize_results.py`

### 环境变量设置
如果使用LLM重排功能，需要设置以下环境变量：
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

Windows用户：
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
```

## 评估指标

- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall@K
- Precision@K
- MRR (Mean Reciprocal Rank)

## 核心技术

### 1. BM25基准模型
传统的信息检索算法，作为性能基准进行比较。

### 2. 密集检索
- 利用Sentence-BERT模型将查询和文档映射到高维向量空间
- 通过余弦相似度计算查询与文档的相关性
- 包括预训练模型和针对医疗领域微调的模型

### 3. LLM重排
利用大语言模型（如通义千问）对初步检索结果进行重排，提升相关性。

### 4. 三阶段混合检索
创新的混合检索架构：
- 第一阶段：BM25进行粗排，筛选出Top-100候选文档
- 第二阶段：微调的Bi-Encoder模型进行精排，选出Top-20
- 第三阶段：LLM对Top-10进行最终重排

## 技术特点

1. **多层次检索架构**：从传统算法到深度学习再到LLM辅助
2. **针对性微调**：针对医疗领域专门微调模型
3. **混合架构设计**：结合多种方法的优势
4. **全面评估体系**：使用多个数据集和指标进行评估
5. **可视化分析**：提供直观的性能对比图表

## 项目目标

本项目的目标是探索和比较不同信息检索方法的效果，特别是在医疗领域文本检索上的性能差异，通过结合传统方法和最新AI技术，寻求最优的检索解决方案。

## 贡献

欢迎贡献以增强检索系统。您可以自由fork仓库，进行改进，并提交pull request。
