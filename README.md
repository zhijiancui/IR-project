# Information Retrieval System Project

This is a deep learning-based information retrieval system project that includes implementations and comparisons of various retrieval methods, including traditional BM25 algorithms, fine-tuned Bi-Encoder models, LLM re-ranking, and a three-stage hybrid retrieval architecture.

## Project Overview

This project aims to improve information retrieval effectiveness in the medical domain through multiple approaches. We implement methods ranging from traditional information retrieval algorithms to state-of-the-art deep learning models, and then to large language model (LLM) assisted retrieval technologies, providing comprehensive evaluation and comparison of these methods.

## Project Structure

### Folder Structure

- `data/` - Stores raw data files
- `datasets/` - Stores processed datasets
- `models/` - Stores trained model files
  - `finetuned-medical-retriever` - Fine-tuned medical domain retrieval model
- `results/` - Stores experimental results and evaluation metrics

### Python Scripts Function Description

- `load_data.py` - Loads NFCorpus medical retrieval dataset
- `download_datasets.py` - Downloads multiple BEIR datasets
- `bm25_baseline.py` - Implements BM25 baseline model
- `dense_retrieval_pretrained.py` - Performs dense retrieval using pretrained models
- `dense_retrieval_finetuned.py` - Performs dense retrieval using fine-tuned models
- `prepare_training_data.py` - Prepares training data
- `train_biencoder_offline.py` - Trains Bi-Encoder model
- `llm_reranking_improved.py` - Uses large language models for re-ranking
- `hybrid_retrieval.py` - Three-stage hybrid retrieval architecture (BM25 coarse ranking + Bi-Encoder fine ranking + LLM re-ranking)
- `multi_dataset_evaluation.py` - Evaluates model performance on multiple datasets
- `visualize_results.py` - Visualizes result comparisons

## Installation

First, clone this repository:
```bash
git clone https://github.com/zhijiancui/IR-project.git
cd IR-project
```

Then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data and Model Files

**Important Note:** Due to GitHub's file size limitations, the data and model files are not included in the repository. You will need to download or generate them separately:

### Dataset Access
The datasets used in this project can be accessed via our shared cloud drive:
- Link：[[IR project](https://bhpan.buaa.edu.cn/link/AAD502EC85886C45818CF89E6957360695)]，password：nvbh
- Content：
  - BEIR benchmark datasets`datasets.zip`
  - Training data for Bi-Encoder model`data.zip`
  - results and statistic figures`results.zip`

After downloading, unzip the files and place them in the following folders:
```
IR project/
├── datasets/
├── data/
└── results/
```

Alternatively, you can use `download_datasets.py` to download some public datasets.

### Model Access
The trained models can be accessed via:
- Link: [[IR project](https://bhpan.buaa.edu.cn/link/AAD502EC85886C45818CF89E6957360695)]，password：nvbh
- Content:
  - Fine-tuned medical retriever model (`models/finetuned-medical-retriever/`)
  - Checkpoint files for different training stages
  - Pre-trained base models

After downloading, place the models in the following folder:
```
models/
└── finetuned-medical-retriever/
    ├── 1_Pooling/
    ├── checkpoints/
    └── [model files]
```

### Model Training
If you prefer to train the models yourself:
1. Prepare training data using `prepare_training_data.py`
2. Train the Bi-Encoder model using `train_biencoder_offline.py`
3. The trained model will be saved to `models/finetuned-medical-retriever/`

## Usage

### Environment Preparation
```bash
pip install -r requirements.txt
```

### Execution Flow
1. Download datasets: Run `download_datasets.py`
2. Run BM25 baseline: Run `bm25_baseline.py`
3. Run fine-tuned model: Run `dense_retrieval_finetuned.py`
4. Run LLM re-ranking: Run `llm_reranking_improved.py`
5. Run hybrid architecture: Run `hybrid_retrieval.py`
6. View visualization results: Run `visualize_results.py`

### Environment Variable Setup
If using LLM re-ranking functionality, set the following environment variable:
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

For Windows users:
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
```

## Evaluation Metrics

- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall@K
- Precision@K
- MRR (Mean Reciprocal Rank)

## Core Technologies

### 1. BM25 Baseline Model
Traditional information retrieval algorithm serving as performance baseline for comparison.

### 2. Dense Retrieval
- Utilizes Sentence-BERT models to map queries and documents to high-dimensional vector space
- Calculates relevance between queries and documents through cosine similarity
- Includes both pretrained models and models fine-tuned for the medical domain

### 3. LLM Re-ranking
Uses large language models (such as Qwen) to re-rank preliminary retrieval results, improving relevance.

### 4. Three-Stage Hybrid Retrieval
Innovative hybrid retrieval architecture:
- Stage 1: BM25 performs coarse ranking, filtering top-100 candidate documents
- Stage 2: Fine-tuned Bi-Encoder model performs fine ranking, selecting top-20
- Stage 3: LLM performs final re-ranking of top-10

## Technical Features

1. **Multi-level retrieval architecture**: From traditional algorithms to deep learning and then to LLM assistance
2. **Targeted fine-tuning**: Models specifically fine-tuned for the medical domain
3. **Hybrid architecture design**: Combines advantages of multiple approaches
4. **Comprehensive evaluation system**: Uses multiple datasets and metrics for evaluation
5. **Visualization analysis**: Provides intuitive performance comparison charts

## Project Goals

The goal of this project is to explore and compare the effectiveness of different information retrieval methods, especially their performance differences in medical domain text retrieval. By combining traditional methods with the latest AI technologies, we seek optimal retrieval solutions.

## Contributing

Contributions to enhance the retrieval system are welcome. Feel free to fork the repository, make improvements, and submit pull requests.
