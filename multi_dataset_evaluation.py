"""
多数据集评估框架
在多个BEIR标准数据集上评估BM25、微调Bi-Encoder、LLM重排和混合架构
"""
import os
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util
from dashscope import Generation
import dashscope


# BEIR数据集配置
DATASETS = [
    'nfcorpus',      # 医学领域 (小规模)
    # 'scifact',       # 科学事实验证 (小规模)
    # 'fiqa',          # 金融问答 (中等规模)
    # 'trec-covid',    # COVID-19文献 (中等规模)
    # 'scidocs',       # 科学文献引用 (中等规模)
]

# 模型配置
BIENCODER_MODEL_PATH = "models/finetuned-medical-retriever"
LLM_MODEL = "qwen-max"

# 测试模式配置
TEST_MODE = True  # 设置为True使用10%子集，False使用完整数据集
TEST_SUBSET_RATIO = 0.1  # 测试模式下使用的数据比例

# 结果保存路径
RESULTS_DIR = Path("results/multi_dataset")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_text(text: str) -> List[str]:
    """文本预处理：转小写并分词"""
    return text.lower().split()


def build_bm25_index(corpus: Dict) -> Tuple[BM25Okapi, List[str]]:
    """构建BM25索引"""
    print("  构建BM25索引...")
    doc_ids = []
    tokenized_corpus = []

    for doc_id, doc_content in corpus.items():
        doc_ids.append(doc_id)
        title = doc_content.get('title', '')
        text = doc_content.get('text', '')
        full_text = f"{title} {text}"
        tokenized_corpus.append(preprocess_text(full_text))

    bm25_model = BM25Okapi(tokenized_corpus)
    return bm25_model, doc_ids


def bm25_search(queries: Dict, corpus: Dict, top_k: int = 100) -> Dict[str, Dict[str, float]]:
    """BM25检索"""
    print("  运行BM25检索...")
    bm25_model, doc_ids = build_bm25_index(corpus)

    results = {}
    for query_id, query_text in queries.items():
        tokenized_query = preprocess_text(query_text)
        scores = bm25_model.get_scores(tokenized_query)

        # 获取top-k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[query_id] = {
            doc_ids[idx]: float(scores[idx]) for idx in top_indices
        }

    return results


def biencoder_search(queries: Dict, corpus: Dict, model_path: str,
                     bm25_results: Dict = None, top_k: int = 100) -> Dict[str, Dict[str, float]]:
    """Bi-Encoder检索（可选基于BM25结果）"""
    print(f"  加载Bi-Encoder模型: {model_path}")
    model = SentenceTransformer(model_path, device='cpu')

    results = {}
    total_queries = len(queries)
    start_time = time.time()

    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        # 如果提供了BM25结果，只对BM25的top-k进行重排
        if bm25_results and query_id in bm25_results:
            candidate_ids = list(bm25_results[query_id].keys())[:top_k]
            candidate_docs = [
                f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
                for doc_id in candidate_ids
            ]
        else:
            # 否则对整个语料库进行检索
            candidate_ids = list(corpus.keys())
            candidate_docs = [
                f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
                for doc_id in candidate_ids
            ]

        # 编码并计算相似度
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        doc_embeddings = model.encode(candidate_docs, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

        # 获取top-k结果
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
        results[query_id] = {
            candidate_ids[idx]: float(similarities[idx].cpu()) for idx in top_indices
        }

        # 显示进度
        if i % 10 == 0 or i == total_queries:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_queries - i) * avg_time
            print(f"    进度: {i}/{total_queries} ({i*100//total_queries}%) | "
                  f"已用时: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s")

    return results


def construct_reranking_prompt(query: str, docs: List[Dict], top_k: int = 10) -> str:
    """构建LLM重排提示词"""
    prompt = """你是一个信息检索专家。你的任务是根据查询对文档进行相关性排序。

【重要规则】
1. 只输出文档编号的排序，不要输出任何解释
2. 格式必须严格为：1,3,5,2,4（用逗号分隔）
3. 不要使用其他符号（如 >、-、.）
4. 不要输出"排序结果："等前缀文字

【示例1】
查询：糖尿病的治疗方法
文档：
[1] 标题：胰岛素治疗2型糖尿病 内容：本文介绍胰岛素在2型糖尿病治疗中的应用...
[2] 标题：心血管疾病预防 内容：心血管疾病是全球主要死因...
[3] 标题：糖尿病饮食管理 内容：合理的饮食控制对糖尿病患者至关重要...

你的输出：1,3,2

【示例2】
查询：高血压的症状
文档：
[1] 标题：糖尿病并发症 内容：糖尿病可能导致多种并发症...
[2] 标题：高血压临床表现 内容：高血压患者常见症状包括头痛、头晕...
[3] 标题：高血压诊断标准 内容：根据WHO标准，收缩压≥140mmHg...

你的输出：2,3,1

【现在开始正式任务】
"""

    prompt += f"\n查询：{query}\n文档：\n"

    for i, doc in enumerate(docs[:top_k], 1):
        title = doc.get('title', '')[:100]
        text = doc.get('text', '')[:200]
        prompt += f"[{i}] 标题：{title} 内容：{text}\n"

    prompt += f"\n请对以上{min(len(docs), top_k)}个文档进行排序，只输出编号序列："
    return prompt


def call_llm_api(prompt: str, model: str = "qwen-max") -> str:
    """调用DashScope API（使用环境变量中的API密钥）"""
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            temperature=0.01,
            top_p=0.5,
            result_format='message'
        )

        if response.status_code == 200:
            if hasattr(response.output, 'choices') and response.output.choices:
                return response.output.choices[0].message.content
            elif hasattr(response.output, 'text') and response.output.text:
                return response.output.text
    except Exception as e:
        print(f"    API调用失败: {e}")

    return ""


def parse_llm_ranking(llm_output: str, num_docs: int) -> List[int]:
    """解析LLM输出的排序"""
    try:
        llm_output = llm_output.strip()
        ranking = [int(x.strip()) for x in llm_output.split(',')]

        # 验证排序
        if len(ranking) == num_docs and set(ranking) == set(range(1, num_docs + 1)):
            return ranking
    except:
        pass

    # 如果解析失败，返回原始顺序
    return list(range(1, num_docs + 1))


def llm_reranking(queries: Dict, corpus: Dict, base_results: Dict,
                  model: str = "qwen-max", top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """LLM重排"""
    print(f"  运行LLM重排 (模型: {model})...")
    results = {}
    total_queries = len(queries)
    start_time = time.time()

    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        if query_id not in base_results:
            continue

        # 获取候选文档
        candidate_ids = list(base_results[query_id].keys())[:20]  # 对top-20进行重排
        candidate_docs = [corpus[doc_id] for doc_id in candidate_ids]

        # 构建提示词并调用LLM
        prompt = construct_reranking_prompt(query_text, candidate_docs, min(len(candidate_docs), top_k))
        llm_output = call_llm_api(prompt, model)

        # 解析排序结果
        ranking = parse_llm_ranking(llm_output, min(len(candidate_docs), top_k))

        # 构建结果
        reranked_ids = [candidate_ids[r - 1] for r in ranking]
        remaining_ids = [doc_id for doc_id in candidate_ids if doc_id not in reranked_ids]
        final_ids = reranked_ids + remaining_ids

        results[query_id] = {
            doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(final_ids)
        }

        # 显示进度
        if i % 10 == 0 or i == total_queries:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_queries - i) * avg_time
            print(f"    进度: {i}/{total_queries} ({i*100//total_queries}%) | "
                  f"已用时: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s")

    return results


def hybrid_retrieval(queries: Dict, corpus: Dict, qrels: Dict,
                     biencoder_path: str, llm_model: str = "qwen-max") -> Dict[str, Dict[str, float]]:
    """三阶段混合检索"""
    print("  运行三阶段混合检索...")

    # Stage 1: BM25粗排 (Top-100)
    print("    Stage 1: BM25粗排...")
    bm25_results = bm25_search(queries, corpus, top_k=100)

    # Stage 2: Bi-Encoder精排 (Top-20)
    print("    Stage 2: Bi-Encoder精排...")
    biencoder_results = biencoder_search(queries, corpus, biencoder_path, bm25_results, top_k=20)

    # Stage 3: LLM重排 (Top-10)
    print("    Stage 3: LLM重排...")
    final_results = llm_reranking(queries, corpus, biencoder_results, llm_model, top_k=10)

    return final_results


def evaluate_dataset(dataset_name: str):
    """评估单个数据集"""
    print(f"\n{'='*60}")
    print(f"评估数据集: {dataset_name}")
    print(f"{'='*60}")

    # 加载数据
    print("加载数据...")
    data_path = f"datasets/{dataset_name}"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # 测试模式：使用子集
    if TEST_MODE:
        import random
        random.seed(42)  # 固定随机种子以保证可重复性

        # 计算子集大小
        subset_size = max(30, int(len(queries) * TEST_SUBSET_RATIO))

        # 随机选择查询子集
        query_ids = list(queries.keys())
        selected_query_ids = random.sample(query_ids, subset_size)

        # 过滤queries和qrels
        queries = {qid: queries[qid] for qid in selected_query_ids}
        qrels = {qid: qrels[qid] for qid in selected_query_ids if qid in qrels}

        print(f"  [测试模式] 使用 {TEST_SUBSET_RATIO*100:.0f}% 子集")

    print(f"  语料库大小: {len(corpus)}")
    print(f"  查询数量: {len(queries)}")
    print(f"  相关判断数量: {sum(len(v) for v in qrels.values())}")

    results = {}

    # 1. BM25基线
    print("\n[1/4] BM25基线")
    start_time = time.time()
    bm25_results = bm25_search(queries, corpus, top_k=100)
    bm25_time = time.time() - start_time

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, bm25_results, [1, 3, 5, 10, 100])
    bm25_metrics = {**ndcg, **_map, **recall, **precision}
    results['bm25'] = {
        'metrics': bm25_metrics,
        'time': bm25_time
    }
    print(f"  NDCG@10: {bm25_metrics['NDCG@10']:.4f}")
    print(f"  耗时: {bm25_time:.2f}秒")

    # 2. 微调Bi-Encoder
    print("\n[2/4] 微调Bi-Encoder")
    if os.path.exists(BIENCODER_MODEL_PATH):
        start_time = time.time()
        # 基于BM25的Top-100结果进行重排，避免对整个语料库编码
        biencoder_results = biencoder_search(queries, corpus, BIENCODER_MODEL_PATH, bm25_results, top_k=100)
        biencoder_time = time.time() - start_time

        ndcg, _map, recall, precision = evaluator.evaluate(qrels, biencoder_results, [1, 3, 5, 10, 100])
        biencoder_metrics = {**ndcg, **_map, **recall, **precision}
        results['biencoder'] = {
            'metrics': biencoder_metrics,
            'time': biencoder_time
        }
        print(f"  NDCG@10: {biencoder_metrics['NDCG@10']:.4f}")
        print(f"  耗时: {biencoder_time:.2f}秒")
    else:
        print(f"  跳过：模型不存在 ({BIENCODER_MODEL_PATH})")
        biencoder_results = None

    # 3. LLM重排（基于BM25）
    print("\n[3/4] LLM重排")
    start_time = time.time()
    llm_results = llm_reranking(queries, corpus, bm25_results, LLM_MODEL, top_k=10)
    llm_time = time.time() - start_time

    ndcg, _map, recall, precision = evaluator.evaluate(qrels, llm_results, [1, 3, 5, 10, 100])
    llm_metrics = {**ndcg, **_map, **recall, **precision}
    results['llm'] = {
        'metrics': llm_metrics,
        'time': llm_time
    }
    print(f"  NDCG@10: {llm_metrics['NDCG@10']:.4f}")
    print(f"  耗时: {llm_time:.2f}秒")

    # 4. 混合架构
    print("\n[4/4] 混合架构")
    if biencoder_results is not None:
        start_time = time.time()
        hybrid_results = hybrid_retrieval(queries, corpus, qrels, BIENCODER_MODEL_PATH, LLM_MODEL)
        hybrid_time = time.time() - start_time

        ndcg, _map, recall, precision = evaluator.evaluate(qrels, hybrid_results, [1, 3, 5, 10, 100])
        hybrid_metrics = {**ndcg, **_map, **recall, **precision}
        results['hybrid'] = {
            'metrics': hybrid_metrics,
            'time': hybrid_time
        }
        print(f"  NDCG@10: {hybrid_metrics['NDCG@10']:.4f}")
        print(f"  耗时: {hybrid_time:.2f}秒")
    else:
        print("  跳过：Bi-Encoder模型不存在")

    # 保存结果
    output_file = RESULTS_DIR / f"{dataset_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

    return results


def generate_summary_report(all_results: Dict):
    """生成汇总报告"""
    print("\n" + "="*60)
    print("生成汇总报告")
    print("="*60)

    report = "# 多数据集评估汇总报告\n\n"
    report += f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # 数据集概览
    report += "## 数据集概览\n\n"
    report += "| 数据集 | 领域 | 语料库大小 | 查询数量 |\n"
    report += "|--------|------|-----------|----------|\n"

    dataset_info = {
        'nfcorpus': ('医学', '3,633', '323'),
        'scifact': ('科学事实', '5,183', '300'),
        'fiqa': ('金融', '57,638', '648'),
        'trec-covid': ('COVID-19', '171,332', '50'),
        'scidocs': ('科学文献', '25,657', '1,000'),
    }

    for dataset in DATASETS:
        if dataset in dataset_info:
            domain, corpus_size, query_count = dataset_info[dataset]
            report += f"| {dataset} | {domain} | {corpus_size} | {query_count} |\n"

    # 性能对比表格
    report += "\n## 性能对比 (NDCG@10)\n\n"
    report += "| 数据集 | BM25 | Bi-Encoder | LLM | 混合架构 |\n"
    report += "|--------|------|-----------|-----|----------|\n"

    for dataset in DATASETS:
        if dataset in all_results:
            res = all_results[dataset]
            bm25_ndcg = res.get('bm25', {}).get('metrics', {}).get('NDCG@10', 0)
            bi_ndcg = res.get('biencoder', {}).get('metrics', {}).get('NDCG@10', 0)
            llm_ndcg = res.get('llm', {}).get('metrics', {}).get('NDCG@10', 0)
            hybrid_ndcg = res.get('hybrid', {}).get('metrics', {}).get('NDCG@10', 0)

            report += f"| {dataset} | {bm25_ndcg:.4f} | {bi_ndcg:.4f} | {llm_ndcg:.4f} | {hybrid_ndcg:.4f} |\n"

    # 平均性能
    report += "\n## 平均性能\n\n"
    avg_metrics = {'bm25': [], 'biencoder': [], 'llm': [], 'hybrid': []}

    for dataset in DATASETS:
        if dataset in all_results:
            res = all_results[dataset]
            for method in avg_metrics.keys():
                if method in res:
                    ndcg = res[method].get('metrics', {}).get('NDCG@10', 0)
                    if ndcg > 0:
                        avg_metrics[method].append(ndcg)

    report += "| 方法 | 平均NDCG@10 | 数据集数量 |\n"
    report += "|------|------------|----------|\n"

    for method, scores in avg_metrics.items():
        if scores:
            avg_score = np.mean(scores)
            report += f"| {method} | {avg_score:.4f} | {len(scores)} |\n"

    # 详细指标
    report += "\n## 详细指标\n\n"

    for dataset in DATASETS:
        if dataset not in all_results:
            continue

        report += f"### {dataset}\n\n"
        res = all_results[dataset]

        for method in ['bm25', 'biencoder', 'llm', 'hybrid']:
            if method not in res:
                continue

            metrics = res[method]['metrics']
            exec_time = res[method]['time']

            report += f"**{method.upper()}**\n"
            report += f"- NDCG@10: {metrics.get('NDCG@10', 0):.4f}\n"
            report += f"- MAP@10: {metrics.get('MAP@10', 0):.4f}\n"
            report += f"- Recall@10: {metrics.get('Recall@10', 0):.4f}\n"
            report += f"- P@1: {metrics.get('P@1', 0):.4f}\n"
            report += f"- MRR@10: {metrics.get('MRR@10', 0):.4f}\n"
            report += f"- 耗时: {exec_time:.2f}秒\n\n"

    # 保存报告
    report_file = RESULTS_DIR / "summary_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"汇总报告已保存到: {report_file}")


def main():
    """主函数"""
    print("="*60)
    print("多数据集评估框架")
    print("="*60)

    # 评估所有数据集
    all_results = {}

    for dataset in DATASETS:
        try:
            results = evaluate_dataset(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"\n错误：评估 {dataset} 时出错: {e}")
            continue

    # 生成汇总报告
    if all_results:
        generate_summary_report(all_results)

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == "__main__":
    main()
