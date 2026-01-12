"""
使用微调后的Bi-Encoder进行密集检索
评估微调模型的性能并与BM25对比
"""
import os
import json
import time
import pathlib
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# 强制禁用CUDA
import torch
torch.cuda.is_available = lambda: False


def load_finetuned_model(model_path: str) -> SentenceTransformer:
    """
    加载微调后的模型

    Args:
        model_path: 模型路径

    Returns:
        model: 加载的模型
    """
    print(f"正在加载微调后的模型: {model_path}")
    model = SentenceTransformer(model_path)
    print(f"模型加载成功！")
    print(f"模型维度: {model.get_sentence_embedding_dimension()}")
    return model


def encode_corpus(model: SentenceTransformer, corpus: Dict,
                  batch_size: int = 32) -> tuple:
    """
    编码文档库

    Args:
        model: SentenceTransformer模型
        corpus: 文档库
        batch_size: 批次大小

    Returns:
        corpus_embeddings: 文档向量
        doc_ids: 文档ID列表
    """
    print("\n正在编码文档库...")
    start_time = time.time()

    doc_ids = []
    doc_texts = []

    for doc_id, doc_content in corpus.items():
        doc_ids.append(doc_id)
        title = doc_content.get('title', '')
        text = doc_content.get('text', '')
        full_text = f"{title} {text}".strip()
        doc_texts.append(full_text)

    # 批量编码
    corpus_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False
    )

    elapsed_time = time.time() - start_time
    print(f"文档编码完成！耗时: {elapsed_time:.2f}秒")
    print(f"文档数量: {len(doc_ids)}")
    print(f"向量维度: {corpus_embeddings.shape[1]}")

    return corpus_embeddings, doc_ids


def encode_queries(model: SentenceTransformer, queries: Dict,
                   batch_size: int = 32) -> tuple:
    """
    编码查询

    Args:
        model: SentenceTransformer模型
        queries: 查询字典
        batch_size: 批次大小

    Returns:
        query_embeddings: 查询向量
        query_ids: 查询ID列表
    """
    print("\n正在编码查询...")
    start_time = time.time()

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False
    )

    elapsed_time = time.time() - start_time
    print(f"查询编码完成！耗时: {elapsed_time:.2f}秒")
    print(f"查询数量: {len(query_ids)}")

    return query_embeddings, query_ids


def semantic_search(query_embeddings: np.ndarray,
                   corpus_embeddings: np.ndarray,
                   query_ids: List[str],
                   doc_ids: List[str],
                   top_k: int = 100) -> Dict:
    """
    语义检索

    Args:
        query_embeddings: 查询向量
        corpus_embeddings: 文档向量
        query_ids: 查询ID列表
        doc_ids: 文档ID列表
        top_k: 返回top-k结果

    Returns:
        results: 检索结果 {query_id: {doc_id: score}}
    """
    print(f"\n正在进行语义检索 (top-{top_k})...")
    start_time = time.time()

    results = {}

    for i, query_id in enumerate(query_ids):
        query_embedding = query_embeddings[i:i+1]

        # 计算相似度
        hits = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=top_k
        )[0]

        # 构建结果字典
        results[query_id] = {}
        for hit in hits:
            doc_id = doc_ids[hit['corpus_id']]
            score = float(hit['score'])
            results[query_id][doc_id] = score

    elapsed_time = time.time() - start_time
    print(f"检索完成！耗时: {elapsed_time:.2f}秒")
    print(f"平均每个查询耗时: {elapsed_time/len(query_ids):.4f}秒")

    return results


def evaluate_results(results: Dict, qrels: Dict) -> Dict:
    """
    评估检索结果

    Args:
        results: 检索结果
        qrels: 标准答案

    Returns:
        metrics: 评估指标
    """
    print("\n正在评估检索结果...")

    evaluator = EvaluateRetrieval()

    # 计算各项指标
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, [1, 3, 5, 10, 100]
    )

    mrr = evaluator.evaluate_custom(qrels, results, [1, 10, 100], metric="mrr")

    metrics = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr
    }

    return metrics


def print_metrics(metrics: Dict, model_name: str):
    """
    打印评估指标

    Args:
        metrics: 评估指标
        model_name: 模型名称
    """
    print("\n" + "="*60)
    print(f"{model_name} 检索性能评估结果")
    print("="*60)

    for metric_name, metric_values in metrics.items():
        print(f"\n{metric_name}:")
        for k, v in sorted(metric_values.items()):
            print(f"  {k}: {v:.4f}")


def compare_with_bm25(finetuned_metrics: Dict, bm25_metrics_file: str):
    """
    与BM25基线对比

    Args:
        finetuned_metrics: 微调模型指标
        bm25_metrics_file: BM25指标文件路径
    """
    if not os.path.exists(bm25_metrics_file):
        print(f"\n警告: 找不到BM25指标文件 {bm25_metrics_file}")
        return

    with open(bm25_metrics_file, 'r', encoding='utf-8') as f:
        bm25_metrics = json.load(f)

    print("\n" + "="*60)
    print("性能对比: 微调模型 vs BM25")
    print("="*60)

    # 对比关键指标
    key_metrics = [
        ('NDCG', 'NDCG@10'),
        ('MAP', 'MAP@10'),
        ('Recall', 'Recall@10'),
        ('Precision', 'P@10'),
        ('MRR', 'MRR@10')
    ]

    print(f"\n{'指标':<15} {'BM25':<12} {'微调模型':<12} {'提升':<12}")
    print("-" * 60)

    for metric_type, metric_key in key_metrics:
        bm25_value = bm25_metrics[metric_type][metric_key]
        finetuned_value = finetuned_metrics[metric_type][metric_key]
        improvement = ((finetuned_value - bm25_value) / bm25_value) * 100

        print(f"{metric_key:<15} {bm25_value:<12.4f} {finetuned_value:<12.4f} {improvement:+.2f}%")


def save_results(results: Dict, metrics: Dict, model_name: str,
                output_dir: str = "results"):
    """
    保存结果

    Args:
        results: 检索结果
        metrics: 评估指标
        model_name: 模型名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存检索结果
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    results_file = os.path.join(output_dir, f"{safe_model_name}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n检索结果已保存到: {results_file}")

    # 保存评估指标
    metrics_file = os.path.join(output_dir, f"{safe_model_name}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {metrics_file}")


def show_example_results(results: Dict, queries: Dict, corpus: Dict,
                        qrels: Dict, num_examples: int = 3):
    """
    显示示例检索结果

    Args:
        results: 检索结果
        queries: 查询字典
        corpus: 文档库
        qrels: 标准答案
        num_examples: 显示的示例数量
    """
    print("\n" + "="*60)
    print("示例检索结果")
    print("="*60)

    query_ids = list(queries.keys())[:num_examples]

    for query_id in query_ids:
        print(f"\n查询 ID: {query_id}")
        print(f"查询文本: {queries[query_id]}")
        print("\nTop-5 检索结果:")

        # 获取top-5结果
        top_results = sorted(
            results[query_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for rank, (doc_id, score) in enumerate(top_results, 1):
            print(f"\n  排名 {rank} (分数: {score:.4f})")
            print(f"  文档 ID: {doc_id}")
            print(f"  标题: {corpus[doc_id].get('title', 'N/A')}")
            print(f"  摘要: {corpus[doc_id]['text'][:150]}...")

            # 检查是否是相关文档
            if query_id in qrels and doc_id in qrels[query_id]:
                print(f"  [+] 相关文档 (相关度: {qrels[query_id][doc_id]})")
            else:
                print(f"  [-] 非相关文档")


def main():
    """
    主函数：使用微调模型进行检索和评估
    """
    print("="*60)
    print("微调Bi-Encoder密集检索实验")
    print("="*60)

    # 1. 加载微调后的模型
    print("\n步骤 1: 加载微调后的模型")
    model_path = "models/finetuned-medical-retriever"
    model = load_finetuned_model(model_path)

    # 2. 加载测试数据
    print("\n步骤 2: 加载NFCorpus测试集")
    data_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "datasets", "nfcorpus"
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(f"文档数量: {len(corpus)}")
    print(f"查询数量: {len(queries)}")

    # 3. 编码文档库
    print("\n步骤 3: 编码文档库")
    corpus_embeddings, doc_ids = encode_corpus(model, corpus, batch_size=32)

    # 4. 编码查询
    print("\n步骤 4: 编码查询")
    query_embeddings, query_ids = encode_queries(model, queries, batch_size=32)

    # 5. 语义检索
    print("\n步骤 5: 执行语义检索")
    results = semantic_search(
        query_embeddings, corpus_embeddings,
        query_ids, doc_ids, top_k=100
    )

    # 6. 评估结果
    print("\n步骤 6: 评估检索性能")
    metrics = evaluate_results(results, qrels)

    # 7. 打印结果
    print_metrics(metrics, "Finetuned Bi-Encoder")

    # 8. 与BM25对比
    print("\n步骤 7: 与BM25基线对比")
    compare_with_bm25(metrics, "results/bm25_metrics.json")

    # 9. 保存结果
    print("\n步骤 8: 保存结果")
    save_results(results, metrics, "finetuned_biencoder")

    # 10. 显示示例结果
    print("\n步骤 9: 显示示例检索结果")
    show_example_results(results, queries, corpus, qrels, num_examples=2)

    print("\n" + "="*60)
    print("微调模型检索实验完成！")
    print("="*60)
    print("\n下一步:")
    print("  1. 运行 visualize_results.py 生成对比可视化")
    print("  2. 分析性能提升的原因")
    print("  3. 撰写实验报告")


if __name__ == "__main__":
    main()
