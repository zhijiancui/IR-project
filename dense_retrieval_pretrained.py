"""
使用预训练Bi-Encoder进行密集检索（无需微调版本）
这个版本直接使用预训练模型进行检索，作为深度学习基线
"""
import os
import json
import time
import pathlib
import numpy as np
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util


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
    print("正在编码文档库...")
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

    # 计算余弦相似度
    # 使用sentence-transformers的util.semantic_search
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
    os.makedirs(output_dir, exist_ok=True

)

    # 保存检索结果
    safe_model_name = model_name.replace('/', '_')
    results_file = os.path.join(output_dir, f"{safe_model_name}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n检索结果已保存到: {results_file}")

    # 保存评估指标
    metrics_file = os.path.join(output_dir, f"{safe_model_name}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {metrics_file}")


def main():
    """
    主函数：使用预训练模型进行密集检索
    """
    print("="*60)
    print("预训练Bi-Encoder密集检索实验")
    print("="*60)

    # 配置
    # 注意：这里我们尝试使用一个较小的模型
    # 如果网络问题，可以尝试使用其他已下载的模型
    model_name = "paraphrase-MiniLM-L3-v2"  # 更小的模型，可能更容易下载

    print(f"\n使用模型: {model_name}")
    print("注意: 如果无法下载模型，请检查网络连接或配置代理")

    try:
        # 1. 加载模型
        print("\n步骤 1: 加载预训练模型")
        print("正在加载模型（首次运行需要下载）...")
        model = SentenceTransformer(model_name)
        print(f"模型加载成功！")
        print(f"模型维度: {model.get_sentence_embedding_dimension()}")

    except Exception as e:
        print(f"\n错误: 无法加载模型")
        print(f"错误信息: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 配置代理（如果在国内）")
        print("3. 手动下载模型文件")
        print("4. 使用本地已有的模型")
        return

    # 2. 加载数据集
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
    print_metrics(metrics, model_name)

    # 8. 保存结果
    print("\n步骤 7: 保存结果")
    save_results(results, metrics, model_name)

    # 9. 与BM25对比
    print("\n" + "="*60)
    print("与BM25基线对比")
    print("="*60)

    bm25_metrics_file = "results/bm25_metrics.json"
    if os.path.exists(bm25_metrics_file):
        with open(bm25_metrics_file, 'r', encoding='utf-8') as f:
            bm25_metrics = json.load(f)

        print("\n性能对比 (NDCG@10):")
        print(f"  BM25:              {bm25_metrics['NDCG']['NDCG@10']:.4f}")
        print(f"  {model_name}: {metrics['NDCG']['NDCG@10']:.4f}")

        improvement = (metrics['NDCG']['NDCG@10'] - bm25_metrics['NDCG']['NDCG@10']) / bm25_metrics['NDCG']['NDCG@10'] * 100
        print(f"  提升: {improvement:+.2f}%")

    print("\n" + "="*60)
    print("密集检索实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
