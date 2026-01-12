"""
三阶段混合检索架构
BM25 粗排 → 微调 Bi-Encoder 精排 → LLM 重排
结合三种方法的优势，达到最佳性能
"""
import os
import json
import time
import pathlib
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util
import torch
import dashscope
from dashscope import Generation

# 强制禁用CUDA，使用CPU
torch.cuda.is_available = lambda: False
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 使用 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_bm25_results(results_file: str) -> Dict:
    """加载 BM25 检索结果"""
    print(f"\n正在加载 BM25 结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"加载完成，共 {len(results)} 个查询")
    return results


def construct_reranking_prompt(query: str, docs: List[Dict], top_k: int = 10) -> str:
    """构造 LLM 重排提示词"""
    prompt = """你是一个医疗信息检索专家。你的任务是根据查询对文档进行相关性排序。

【重要规则】
1. 只输出文档编号的排序，不要输出任何解释
2. 格式必须严格为：1,3,5,2,4（用逗号分隔）
3. 不要使用其他符号（如 >、-、.）
4. 不要输出"排序结果："等前缀文字

【示例1】
查询：糖尿病的治疗方法
文档：
[1] 标题：胰岛素治疗2型糖尿病 内容：本文介绍胰岛素在2型糖尿病治疗中的应用...
[2] 标题：心血管疾病预防 内容：心血管疾病的预防措施包括...
[3] 标题：糖尿病饮食管理 内容：糖尿病患者的饮食控制方法...

你的输出：1,3,2

【示例2】
查询：高血压的症状
文档：
[1] 标题：糖尿病并发症 内容：糖尿病的常见并发症...
[2] 标题：高血压临床表现 内容：高血压患者常见症状包括头痛、头晕...
[3] 标题：高血压诊断标准 内容：高血压的诊断依据...

你的输出：2,3,1

【现在开始正式任务】
"""
    prompt += f"查询：{query}\n\n"
    prompt += "文档：\n"
    for i, doc in enumerate(docs[:top_k], 1):
        title = doc.get('title', 'No title')
        text = doc.get('text', '')
        text_snippet = text[:150] + "..." if len(text) > 150 else text
        prompt += f"[{i}] 标题：{title} 内容：{text_snippet}\n"

    prompt += "\n【请严格按照格式输出】\n"
    prompt += "你的输出："
    return prompt


def parse_llm_ranking(llm_output: str, num_docs: int) -> List[int]:
    """解析 LLM 输出的排序结果"""
    import re
    output = llm_output.strip()

    # 移除可能的前缀
    prefixes = ['你的输出：', '排序结果：', '输出：', '结果：', '排序：']
    for prefix in prefixes:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()

    # 方式1: 逗号分隔
    if ',' in output:
        try:
            first_line = output.split('\n')[0].strip()
            cleaned = re.sub(r'[^\d,]', '', first_line)
            numbers = [int(x) for x in cleaned.split(',') if x]
            if all(1 <= n <= num_docs for n in numbers):
                seen = set()
                ranking = [x for x in numbers if not (x in seen or seen.add(x))]
                missing = [i for i in range(1, num_docs + 1) if i not in ranking]
                ranking.extend(missing)
                return ranking[:num_docs]
        except:
            pass

    # 方式2: 提取所有数字
    all_numbers = re.findall(r'\d+', output)
    if all_numbers:
        try:
            numbers = [int(x) for x in all_numbers if 1 <= int(x) <= num_docs]
            if numbers:
                seen = set()
                ranking = [x for x in numbers if not (x in seen or seen.add(x))]
                missing = [i for i in range(1, num_docs + 1) if i not in ranking]
                ranking.extend(missing)
                return ranking[:num_docs]
        except:
            pass

    # 解析失败，返回原始顺序
    return list(range(1, num_docs + 1))


def call_llm_api(prompt: str, api_key: str, model: str = "qwen-max") -> str:
    """调用 LLM API"""
    dashscope.api_key = api_key

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
        return ""
    except Exception as e:
        print(f"    LLM API 调用异常: {str(e)}")
        return ""


def hybrid_retrieval(queries: Dict, corpus: Dict, qrels: Dict,
                     bm25_results: Dict, biencoder_model_path: str,
                     api_key: str, llm_model: str = "qwen-max",
                     stage1_top_k: int = 100,
                     stage2_top_k: int = 20,
                     stage3_top_k: int = 10,
                     max_queries: int = None) -> Dict:
    """
    三阶段混合检索

    Args:
        queries: 查询字典
        corpus: 文档库
        qrels: 标准答案
        bm25_results: BM25 检索结果
        biencoder_model_path: 微调 Bi-Encoder 模型路径
        api_key: LLM API 密钥
        llm_model: LLM 模型名称
        stage1_top_k: 阶段1（BM25）保留的文档数
        stage2_top_k: 阶段2（Bi-Encoder）保留的文档数
        stage3_top_k: 阶段3（LLM）重排的文档数
        max_queries: 最大处理查询数

    Returns:
        final_results: 最终检索结果
    """
    print("\n" + "="*80)
    print("三阶段混合检索架构")
    print("="*80)
    print(f"\n阶段1: BM25 粗排（Top-{stage1_top_k}）")
    print(f"阶段2: 微调 Bi-Encoder 精排（Top-{stage2_top_k}）")
    print(f"阶段3: LLM 重排（Top-{stage3_top_k}）")
    print(f"\n总查询数: {len(queries)}")

    if max_queries:
        print(f"限制处理前 {max_queries} 个查询（测试模式）")
        query_ids = list(queries.keys())[:max_queries]
    else:
        query_ids = list(queries.keys())

    # 加载微调 Bi-Encoder 模型
    print(f"\n正在加载微调 Bi-Encoder 模型: {biencoder_model_path}")
    biencoder = SentenceTransformer(biencoder_model_path, device='cpu')
    print("模型加载完成")

    final_results = {}
    stage_times = {'stage1': 0, 'stage2': 0, 'stage3': 0}
    total_start = time.time()

    for idx, query_id in enumerate(query_ids, 1):
        query_text = queries[query_id]

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(query_ids)}] 处理查询: {query_id}")
        print(f"查询文本: {query_text[:80]}...")
        print(f"{'='*80}")

        # ========== 阶段1: BM25 粗排 ==========
        stage1_start = time.time()
        bm25_top = bm25_results.get(query_id, {})
        if not bm25_top:
            print("  警告: 没有 BM25 结果，跳过")
            continue

        # 获取 Top-K 文档
        sorted_bm25 = sorted(bm25_top.items(), key=lambda x: x[1], reverse=True)[:stage1_top_k]
        stage1_doc_ids = [doc_id for doc_id, _ in sorted_bm25]

        stage1_time = time.time() - stage1_start
        stage_times['stage1'] += stage1_time
        print(f"\n阶段1 完成: BM25 粗排 Top-{len(stage1_doc_ids)}")
        print(f"  耗时: {stage1_time:.3f}秒")

        # ========== 阶段2: Bi-Encoder 精排 ==========
        stage2_start = time.time()

        # 准备文档文本
        stage1_docs = []
        for doc_id in stage1_doc_ids:
            if doc_id in corpus:
                doc_text = f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
                stage1_docs.append({'id': doc_id, 'text': doc_text})

        if not stage1_docs:
            print("  警告: 没有有效文档，跳过")
            continue

        # 编码查询和文档
        query_embedding = biencoder.encode(query_text, convert_to_tensor=True)
        doc_texts = [doc['text'] for doc in stage1_docs]
        doc_embeddings = biencoder.encode(doc_texts, convert_to_tensor=True, batch_size=32)

        # 计算相似度
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

        # 排序并获取 Top-K
        top_indices = torch.argsort(similarities, descending=True)[:stage2_top_k].tolist()
        stage2_doc_ids = [stage1_docs[i]['id'] for i in top_indices]

        stage2_time = time.time() - stage2_start
        stage_times['stage2'] += stage2_time
        print(f"\n阶段2 完成: Bi-Encoder 精排 Top-{len(stage2_doc_ids)}")
        print(f"  耗时: {stage2_time:.3f}秒")

        # ========== 阶段3: LLM 重排 ==========
        stage3_start = time.time()

        # 准备 LLM 输入
        stage2_docs = []
        for doc_id in stage2_doc_ids[:stage3_top_k]:
            if doc_id in corpus:
                stage2_docs.append({
                    'id': doc_id,
                    'title': corpus[doc_id].get('title', ''),
                    'text': corpus[doc_id].get('text', '')
                })

        if not stage2_docs:
            print("  警告: 没有有效文档，跳过")
            continue

        # 构造 Prompt 并调用 LLM
        prompt = construct_reranking_prompt(query_text, stage2_docs, stage3_top_k)
        llm_output = call_llm_api(prompt, api_key, llm_model)

        if llm_output:
            # 解析 LLM 输出
            ranking = parse_llm_ranking(llm_output, min(stage3_top_k, len(stage2_docs)))
            stage3_doc_ids = [stage2_docs[i-1]['id'] for i in ranking if i <= len(stage2_docs)]

            # 添加剩余文档（阶段2中未被 LLM 重排的）
            remaining_ids = [doc_id for doc_id in stage2_doc_ids if doc_id not in stage3_doc_ids]
            stage3_doc_ids.extend(remaining_ids)
        else:
            # LLM 调用失败，使用阶段2的结果
            print("  LLM 调用失败，使用阶段2结果")
            stage3_doc_ids = stage2_doc_ids

        stage3_time = time.time() - stage3_start
        stage_times['stage3'] += stage3_time
        print(f"\n阶段3 完成: LLM 重排 Top-{stage3_top_k}")
        print(f"  LLM 输出: {llm_output[:100] if llm_output else 'N/A'}...")
        print(f"  耗时: {stage3_time:.3f}秒")

        # 构造最终结果（分数递减）
        final_results[query_id] = {}
        base_score = 100.0
        for rank, doc_id in enumerate(stage3_doc_ids):
            final_results[query_id][doc_id] = base_score - rank * 0.1

        # 显示进度
        total_elapsed = time.time() - total_start
        avg_time = total_elapsed / idx
        remaining_time = avg_time * (len(query_ids) - idx)

        print(f"\n进度统计:")
        print(f"  已完成: {idx}/{len(query_ids)}")
        print(f"  平均耗时: {avg_time:.2f}秒/查询")
        print(f"  预计剩余时间: {remaining_time/60:.1f}分钟")
        print(f"  阶段耗时: Stage1={stage1_time:.3f}s, Stage2={stage2_time:.3f}s, Stage3={stage3_time:.3f}s")

    # 总结
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print("混合检索完成！")
    print(f"{'='*80}")
    print(f"\n总耗时: {total_time/60:.2f}分钟")
    print(f"平均耗时: {total_time/len(query_ids):.2f}秒/查询")
    print(f"\n各阶段平均耗时:")
    print(f"  阶段1 (BM25): {stage_times['stage1']/len(query_ids):.3f}秒/查询")
    print(f"  阶段2 (Bi-Encoder): {stage_times['stage2']/len(query_ids):.3f}秒/查询")
    print(f"  阶段3 (LLM): {stage_times['stage3']/len(query_ids):.3f}秒/查询")

    return final_results


def evaluate_results(results: Dict, qrels: Dict) -> Dict:
    """评估检索结果"""
    print("\n正在评估检索结果...")
    evaluator = EvaluateRetrieval()

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
    """打印评估指标"""
    print("\n" + "="*60)
    print(f"{model_name} 检索性能评估结果")
    print("="*60)

    for metric_name, metric_values in metrics.items():
        print(f"\n{metric_name}:")
        for k, v in sorted(metric_values.items()):
            print(f"  {k}: {v:.4f}")


def compare_all_methods(hybrid_metrics: Dict,
                       bm25_metrics_file: str,
                       finetuned_metrics_file: str,
                       llm_metrics_file: str):
    """对比四种方法的性能"""
    # 加载其他方法的指标
    with open(bm25_metrics_file, 'r', encoding='utf-8') as f:
        bm25_metrics = json.load(f)

    with open(finetuned_metrics_file, 'r', encoding='utf-8') as f:
        finetuned_metrics = json.load(f)

    with open(llm_metrics_file, 'r', encoding='utf-8') as f:
        llm_metrics = json.load(f)

    print("\n" + "="*100)
    print("四种方法性能对比")
    print("="*100)

    # 对比关键指标
    key_metrics = [
        ('NDCG', 'NDCG@10'),
        ('MAP', 'MAP@10'),
        ('Recall', 'Recall@10'),
        ('Precision', 'P@10'),
        ('MRR', 'MRR@10')
    ]

    print(f"\n{'指标':<15} {'BM25':<12} {'微调模型':<12} {'LLM重排':<12} {'混合架构':<12} {'最佳方法':<12}")
    print("-" * 100)

    for metric_type, metric_key in key_metrics:
        bm25_value = bm25_metrics[metric_type][metric_key]
        finetuned_value = finetuned_metrics[metric_type][metric_key]
        llm_value = llm_metrics[metric_type][metric_key]
        hybrid_value = hybrid_metrics[metric_type][metric_key]

        # 找出最佳方法
        values = {
            'BM25': bm25_value,
            '微调模型': finetuned_value,
            'LLM重排': llm_value,
            '混合架构': hybrid_value
        }
        best_method = max(values, key=values.get)

        print(f"{metric_key:<15} {bm25_value:<12.4f} {finetuned_value:<12.4f} {llm_value:<12.4f} {hybrid_value:<12.4f} {best_method:<12}")

    # 计算提升百分比
    print("\n" + "="*100)
    print("相对 BM25 的提升百分比")
    print("="*100)
    print(f"\n{'指标':<15} {'微调模型':<20} {'LLM重排':<20} {'混合架构':<20}")
    print("-" * 100)

    for metric_type, metric_key in key_metrics:
        bm25_value = bm25_metrics[metric_type][metric_key]
        finetuned_value = finetuned_metrics[metric_type][metric_key]
        llm_value = llm_metrics[metric_type][metric_key]
        hybrid_value = hybrid_metrics[metric_type][metric_key]

        finetuned_improvement = ((finetuned_value - bm25_value) / bm25_value) * 100
        llm_improvement = ((llm_value - bm25_value) / bm25_value) * 100
        hybrid_improvement = ((hybrid_value - bm25_value) / bm25_value) * 100

        print(f"{metric_key:<15} {finetuned_improvement:+.2f}%{' '*13} {llm_improvement:+.2f}%{' '*13} {hybrid_improvement:+.2f}%")


def save_results(results: Dict, metrics: Dict, output_dir: str = "results"):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存检索结果
    results_file = os.path.join(output_dir, "hybrid_retrieval_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n检索结果已保存到: {results_file}")

    # 保存评估指标
    metrics_file = os.path.join(output_dir, "hybrid_retrieval_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {metrics_file}")


def main():
    """主函数：三阶段混合检索"""
    print("="*80)
    print("三阶段混合检索架构实验")
    print("BM25 粗排 → 微调 Bi-Encoder 精排 → LLM 重排")
    print("="*80)

    # 配置参数
    STAGE1_TOP_K = 100  # BM25 保留 Top-100
    STAGE2_TOP_K = 20   # Bi-Encoder 精排到 Top-20
    STAGE3_TOP_K = 10   # LLM 重排 Top-10
    LLM_MODEL = "qwen-max"
    MAX_QUERIES = None  # None 表示处理所有查询

    print(f"\n配置参数:")
    print(f"  - 阶段1 (BM25): Top-{STAGE1_TOP_K}")
    print(f"  - 阶段2 (Bi-Encoder): Top-{STAGE2_TOP_K}")
    print(f"  - 阶段3 (LLM): Top-{STAGE3_TOP_K}")
    print(f"  - LLM 模型: {LLM_MODEL}")
    print(f"  - 预计耗时: 10-15 分钟")
    print(f"  - 预计成本: 2-4 元人民币")

    # 获取 API Key
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("\n错误: 未找到 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量:")
        print("  Windows PowerShell: $env:DASHSCOPE_API_KEY='your_api_key'")
        return

    print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")

    # 1. 加载数据
    print("\n步骤 1: 加载 NFCorpus 数据集")
    data_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "datasets", "nfcorpus"
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"文档数量: {len(corpus)}")
    print(f"查询数量: {len(queries)}")

    # 2. 加载 BM25 结果
    print("\n步骤 2: 加载 BM25 检索结果")
    bm25_results = load_bm25_results("results/bm25_results.json")

    # 3. 混合检索
    print("\n步骤 3: 执行三阶段混合检索")
    hybrid_results = hybrid_retrieval(
        queries, corpus, qrels, bm25_results,
        biencoder_model_path="models/finetuned-medical-retriever",
        api_key=api_key,
        llm_model=LLM_MODEL,
        stage1_top_k=STAGE1_TOP_K,
        stage2_top_k=STAGE2_TOP_K,
        stage3_top_k=STAGE3_TOP_K,
        max_queries=MAX_QUERIES
    )

    # 4. 评估结果
    print("\n步骤 4: 评估混合检索性能")
    metrics = evaluate_results(hybrid_results, qrels)

    # 5. 打印结果
    print_metrics(metrics, "混合检索架构 (BM25 + Bi-Encoder + LLM)")

    # 6. 四方对比
    print("\n步骤 5: 四方对比分析")
    compare_all_methods(
        metrics,
        "results/bm25_metrics.json",
        "results/finetuned_biencoder_metrics.json",
        "results/llm_reranking_improved_metrics.json"
    )

    # 7. 保存结果
    print("\n步骤 6: 保存结果")
    save_results(hybrid_results, metrics)

    print("\n" + "="*80)
    print("混合检索实验完成！")
    print("="*80)
    print("\n主要优势:")
    print("  1. 结合三种方法的优势")
    print("  2. BM25 快速粗排")
    print("  3. Bi-Encoder 高质量精排")
    print("  4. LLM 优化 Top-10 排序")


if __name__ == "__main__":
    main()
