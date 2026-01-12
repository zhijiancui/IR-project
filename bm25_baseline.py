"""
BM25基线模型实现
使用传统的BM25算法进行信息检索，作为深度学习模型的对比基线
"""
import os
import json
import time
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import pathlib


def preprocess_text(text: str) -> List[str]:
    """
    简单的文本预处理：转小写并分词

    Args:
        text: 输入文本

    Returns:
        分词后的token列表
    """
    return text.lower().split()


def build_bm25_index(corpus: Dict) -> Tuple[BM25Okapi, List[str]]:
    """
    构建BM25索引

    Args:
        corpus: 文档库字典 {doc_id: {'title': ..., 'text': ...}}

    Returns:
        bm25_model: BM25模型
        doc_ids: 文档ID列表（与BM25索引对应）
    """
    print("正在构建BM25索引...")
    start_time = time.time()

    # 准备文档
    doc_ids = []
    tokenized_corpus = []

    for doc_id, doc_content in corpus.items():
        doc_ids.append(doc_id)
        # 合并标题和正文
        title = doc_content.get('title', '')
        text = doc_content.get('text', '')
        full_text = f"{title} {text}"
        # 分词
        tokens = preprocess_text(full_text)
        tokenized_corpus.append(tokens)

    # 构建BM25索引
    bm25_model = BM25Okapi(tokenized_corpus)

    elapsed_time = time.time() - start_time
    print(f"BM25索引构建完成！耗时: {elapsed_time:.2f}秒")
    print(f"索引文档数: {len(doc_ids)}")

    return bm25_model, doc_ids


def search_bm25(bm25_model: BM25Okapi, doc_ids: List[str],
                queries: Dict, top_k: int = 100) -> Dict:
    """
    使用BM25进行检索

    Args:
        bm25_model: BM25模型
        doc_ids: 文档ID列表
        queries: 查询字典 {query_id: query_text}
        top_k: 返回top-k个结果

    Returns:
        results: 检索结果 {query_id: {doc_id: score}}
    """
    print(f"\n正在使用BM25检索 {len(queries)} 个查询...")
    start_time = time.time()

    results = {}

    for query_id, query_text in queries.items():
        # 分词
        tokenized_query = preprocess_text(query_text)

        # BM25检索
        scores = bm25_model.get_scores(tokenized_query)

        # 获取top-k结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # 构建结果字典
        results[query_id] = {}
        for idx in top_indices:
            doc_id = doc_ids[idx]
            score = float(scores[idx])
            results[query_id][doc_id] = score

    elapsed_time = time.time() - start_time
    print(f"检索完成！耗时: {elapsed_time:.2f}秒")
    print(f"平均每个查询耗时: {elapsed_time/len(queries):.4f}秒")

    return results


def evaluate_results(results: Dict, qrels: Dict) -> Dict:
    """
    评估检索结果

    Args:
        results: 检索结果 {query_id: {doc_id: score}}
        qrels: 标准答案 {query_id: {doc_id: relevance}}

    Returns:
        metrics: 评估指标字典
    """
    print("\n正在评估检索结果...")

    # 使用BEIR的评估工具
    evaluator = EvaluateRetrieval()

    # 计算NDCG, MAP, Recall, Precision
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])

    # 计算MRR (Mean Reciprocal Rank)
    mrr = evaluator.evaluate_custom(qrels, results, [1, 10, 100], metric="mrr")

    # 整合所有指标
    metrics = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr
    }

    return metrics


def print_metrics(metrics: Dict):
    """
    打印评估指标

    Args:
        metrics: 评估指标字典
    """
    print("\n" + "="*60)
    print("BM25 检索性能评估结果")
    print("="*60)

    for metric_name, metric_values in metrics.items():
        print(f"\n{metric_name}:")
        for k, v in sorted(metric_values.items()):
            print(f"  {k}: {v:.4f}")


def save_results(results: Dict, metrics: Dict, output_dir: str = "results"):
    """
    保存检索结果和评估指标

    Args:
        results: 检索结果
        metrics: 评估指标
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存检索结果
    results_file = os.path.join(output_dir, "bm25_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n检索结果已保存到: {results_file}")

    # 保存评估指标
    metrics_file = os.path.join(output_dir, "bm25_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {metrics_file}")


def main():
    """
    主函数：运行BM25基线实验
    """
    print("="*60)
    print("BM25 基线模型实验")
    print("="*60)

    # 1. 加载数据集
    print("\n步骤 1: 加载NFCorpus数据集")
    data_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets", "nfcorpus")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(f"文档数量: {len(corpus)}")
    print(f"查询数量: {len(queries)}")
    print(f"相关性标注数量: {len(qrels)}")

    # 2. 构建BM25索引
    print("\n步骤 2: 构建BM25索引")
    bm25_model, doc_ids = build_bm25_index(corpus)

    # 3. 执行检索
    print("\n步骤 3: 执行BM25检索")
    results = search_bm25(bm25_model, doc_ids, queries, top_k=100)

    # 4. 评估结果
    print("\n步骤 4: 评估检索性能")
    metrics = evaluate_results(results, qrels)

    # 5. 打印结果
    print_metrics(metrics)

    # 6. 保存结果
    print("\n步骤 5: 保存结果")
    save_results(results, metrics)

    # 7. 显示一些示例结果
    print("\n" + "="*60)
    print("示例检索结果")
    print("="*60)

    # 显示第一个查询的top-5结果
    first_query_id = list(queries.keys())[0]
    print(f"\n查询 ID: {first_query_id}")
    print(f"查询文本: {queries[first_query_id]}")
    print("\nTop-5 检索结果:")

    top_results = sorted(results[first_query_id].items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (doc_id, score) in enumerate(top_results, 1):
        print(f"\n排名 {rank} (分数: {score:.4f})")
        print(f"文档 ID: {doc_id}")
        print(f"标题: {corpus[doc_id].get('title', 'N/A')}")
        print(f"摘要: {corpus[doc_id]['text'][:150]}...")

        # 检查是否是相关文档
        if first_query_id in qrels and doc_id in qrels[first_query_id]:
            print(f"[+] 相关文档 (相关度: {qrels[first_query_id][doc_id]})")
        else:
            print("[-] 非相关文档")

    print("\n" + "="*60)
    print("BM25基线实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
