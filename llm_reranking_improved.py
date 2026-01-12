"""
使用阿里云百炼 API 进行 LLM 重排（改进的 Listwise 方法）
改进的 Prompt 设计，使用中文 + Few-shot + 更强的约束
"""
import os
import json
import time
import pathlib
import re
from typing import Dict, List, Tuple
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import dashscope
from dashscope import Generation


def load_bm25_results(results_file: str) -> Dict:
    """
    加载 BM25 检索结果

    Args:
        results_file: BM25 结果文件路径

    Returns:
        results: 检索结果字典
    """
    print(f"\n正在加载 BM25 结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"加载完成，共 {len(results)} 个查询")
    return results


def construct_reranking_prompt_improved(query: str, docs: List[Dict], top_k: int = 20) -> str:
    """
    构造改进的重排提示词（中文 + Few-shot + 强约束）

    Args:
        query: 查询文本
        docs: 文档列表（包含 id, title, text）
        top_k: 使用前 k 个文档

    Returns:
        prompt: 构造的提示词
    """
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

    # 添加查询
    prompt += f"查询：{query}\n\n"

    # 添加文档列表
    prompt += "文档：\n"
    for i, doc in enumerate(docs[:top_k], 1):
        title = doc.get('title', 'No title')
        text = doc.get('text', '')
        # 截断文本
        text_snippet = text[:150] + "..." if len(text) > 150 else text
        prompt += f"[{i}] 标题：{title} 内容：{text_snippet}\n"

    # 强调输出格式
    prompt += "\n【请严格按照格式输出】\n"
    prompt += "你的输出："

    return prompt


def parse_llm_ranking_improved(llm_output: str, num_docs: int) -> List[int]:
    """
    解析 LLM 输出的排序结果（改进版，支持多种格式）

    Args:
        llm_output: LLM 的输出文本
        num_docs: 文档数量

    Returns:
        ranking: 排序后的文档索引列表（从1开始）
    """
    # 清理输出
    output = llm_output.strip()

    # 移除可能的前缀
    prefixes = ['你的输出：', '排序结果：', '输出：', '结果：', '排序：']
    for prefix in prefixes:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()

    # 方式1: 逗号分隔 "1,3,5,2,4"
    if ',' in output:
        try:
            # 提取第一行（如果有多行）
            first_line = output.split('\n')[0].strip()
            # 移除所有非数字和逗号的字符
            cleaned = re.sub(r'[^\d,]', '', first_line)
            numbers = [int(x) for x in cleaned.split(',') if x]
            if all(1 <= n <= num_docs for n in numbers):
                # 去重并保持顺序
                seen = set()
                ranking = [x for x in numbers if not (x in seen or seen.add(x))]
                # 补充缺失的文档编号
                missing = [i for i in range(1, num_docs + 1) if i not in ranking]
                ranking.extend(missing)
                print(f"    成功解析（逗号格式）: {ranking[:5]}...")
                return ranking[:num_docs]
        except Exception as e:
            print(f"    逗号格式解析失败: {e}")

    # 方式2: [3] > [1] > [7] 格式
    matches = re.findall(r'\[(\d+)\]', output)
    if matches:
        try:
            ranking = [int(m) for m in matches if 1 <= int(m) <= num_docs]
            seen = set()
            ranking = [x for x in ranking if not (x in seen or seen.add(x))]
            missing = [i for i in range(1, num_docs + 1) if i not in ranking]
            ranking.extend(missing)
            print(f"    成功解析（方括号格式）: {ranking[:5]}...")
            return ranking[:num_docs]
        except:
            pass

    # 方式3: 空格分隔 "1 3 5 2 4"
    try:
        numbers = [int(x) for x in re.findall(r'\d+', output.split('\n')[0])]
        if numbers and all(1 <= n <= num_docs for n in numbers):
            seen = set()
            ranking = [x for x in numbers if not (x in seen or seen.add(x))]
            missing = [i for i in range(1, num_docs + 1) if i not in ranking]
            ranking.extend(missing)
            print(f"    成功解析（空格格式）: {ranking[:5]}...")
            return ranking[:num_docs]
    except:
        pass

    # 方式4: 提取所有数字
    all_numbers = re.findall(r'\d+', output)
    if all_numbers:
        try:
            numbers = [int(x) for x in all_numbers if 1 <= int(x) <= num_docs]
            if numbers:
                seen = set()
                ranking = [x for x in numbers if not (x in seen or seen.add(x))]
                missing = [i for i in range(1, num_docs + 1) if i not in ranking]
                ranking.extend(missing)
                print(f"    成功解析（提取数字）: {ranking[:5]}...")
                return ranking[:num_docs]
        except:
            pass

    # 所有方式都失败，返回原始顺序
    print(f"    警告: 无法解析 LLM 输出，使用原始顺序")
    print(f"    原始输出: {output[:100]}...")
    return list(range(1, num_docs + 1))


def call_dashscope_api(prompt: str, api_key: str, model: str = "qwen-plus") -> str:
    """
    调用阿里云百炼 API

    Args:
        prompt: 提示词
        api_key: API 密钥
        model: 模型名称 (qwen-plus, qwen-turbo, qwen-max)

    Returns:
        response_text: API 返回的文本
    """
    dashscope.api_key = api_key

    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            temperature=0.01,  # 极低温度以获得最确定的输出
            top_p=0.5,
            result_format='message'
        )

        if response.status_code == 200:
            # 正确的访问方式：response.output.choices[0].message.content
            if hasattr(response.output, 'choices') and response.output.choices:
                return response.output.choices[0].message.content
            elif hasattr(response.output, 'text') and response.output.text:
                return response.output.text
            else:
                print(f"    无法解析响应结构")
                return ""
        else:
            print(f"    API 调用失败: {response.code} - {response.message}")
            return ""

    except Exception as e:
        print(f"    API 调用异常: {str(e)}")
        return ""


def llm_rerank_query(api_key: str, query: str, top_docs: List[Dict],
                     top_k: int = 20, model: str = "qwen-plus") -> List[str]:
    """
    使用 LLM API 对单个查询的 Top-K 文档进行重排

    Args:
        api_key: API 密钥
        query: 查询文本
        top_docs: Top-K 文档列表
        top_k: 重排的文档数量
        model: 模型名称

    Returns:
        reranked_doc_ids: 重排后的文档 ID 列表
    """
    # 构造改进的提示词
    prompt = construct_reranking_prompt_improved(query, top_docs, top_k)

    # 调用 API
    generated_text = call_dashscope_api(prompt, api_key, model)

    if not generated_text:
        # API 调用失败，返回原始顺序
        print("API 调用失败，返回原始顺序")
        return [doc['id'] for doc in top_docs]
    print(f"LLM 输出: {generated_text[:100]}...")
    # 解析排序
    ranking = parse_llm_ranking_improved(generated_text, min(top_k, len(top_docs)))
    print(f"排序结果: {ranking[:5]}...")
    # 根据排序重新组织文档 ID
    reranked_doc_ids = [top_docs[i-1]['id'] for i in ranking if i <= len(top_docs)]

    # 添加剩余文档（如果有超过 top_k 的）
    if len(top_docs) > top_k:
        remaining_ids = [doc['id'] for doc in top_docs[top_k:]]
        reranked_doc_ids.extend(remaining_ids)

    return reranked_doc_ids


def llm_rerank_all(api_key: str, queries: Dict, corpus: Dict,
                   bm25_results: Dict, top_k: int = 20,
                   model: str = "qwen-plus",
                   max_queries: int = None) -> Dict:
    """
    使用 LLM API 对所有查询进行重排

    Args:
        api_key: API 密钥
        queries: 查询字典
        corpus: 文档库
        bm25_results: BM25 检索结果
        top_k: 重排的文档数量
        model: 模型名称
        max_queries: 最大处理查询数（用于测试）

    Returns:
        reranked_results: 重排后的结果 {query_id: {doc_id: score}}
    """
    print(f"\n开始 LLM 重排（改进的 Listwise 方法，Top-{top_k}）...")
    print(f"使用模型: {model}")
    print(f"总查询数: {len(queries)}")

    if max_queries:
        print(f"限制处理前 {max_queries} 个查询（测试模式）")
        query_ids = list(queries.keys())[:max_queries]
    else:
        query_ids = list(queries.keys())

    reranked_results = {}
    total_time = 0
    success_count = 0
    fail_count = 0

    for idx, query_id in enumerate(query_ids, 1):
        query_text = queries[query_id]

        # 获取 BM25 的 Top-K 结果
        bm25_top = bm25_results.get(query_id, {})
        if not bm25_top:
            print(f"  警告: 查询 {query_id} 没有 BM25 结果，跳过")
            continue

        # 按分数排序获取 Top-K 文档
        sorted_docs = sorted(bm25_top.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 构造文档列表
        top_docs = []
        for doc_id, score in sorted_docs:
            if doc_id in corpus:
                top_docs.append({
                    'id': doc_id,
                    'title': corpus[doc_id].get('title', ''),
                    'text': corpus[doc_id].get('text', '')
                })

        if not top_docs:
            print(f"  警告: 查询 {query_id} 没有有效文档，跳过")
            continue

        # LLM 重排
        print(f"\n[{idx}/{len(query_ids)}] 处理查询: {query_id}")
        print(f"  查询文本: {query_text[:80]}...")

        start_time = time.time()
        reranked_doc_ids = llm_rerank_query(api_key, query_text, top_docs, top_k, model)
        elapsed = time.time() - start_time
        total_time += elapsed

        if reranked_doc_ids:
            success_count += 1
        else:
            fail_count += 1

        print(f"  重排完成，耗时: {elapsed:.2f}秒")
        print(f"  平均耗时: {total_time/idx:.2f}秒/查询")
        print(f"  预计剩余时间: {(total_time/idx) * (len(query_ids) - idx) / 60:.1f}分钟")
        print(f"  成功/失败: {success_count}/{fail_count}")

        # 构造结果（分数递减）
        reranked_results[query_id] = {}
        base_score = 100.0
        for rank, doc_id in enumerate(reranked_doc_ids):
            reranked_results[query_id][doc_id] = base_score - rank * 0.1

        # 添加 BM25 中剩余的文档（排在重排结果之后）
        for doc_id, bm25_score in sorted_docs[top_k:]:
            if doc_id not in reranked_results[query_id]:
                reranked_results[query_id][doc_id] = bm25_score * 0.5  # 降低分数

        # 每 10 个查询休息一下，避免 API 限流
        if idx % 10 == 0:
            time.sleep(1)

    print(f"\n重排完成！")
    print(f"总耗时: {total_time/60:.2f}分钟")
    print(f"平均耗时: {total_time/len(query_ids):.2f}秒/查询")
    print(f"成功: {success_count}, 失败: {fail_count}")

    return reranked_results


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


def compare_all_methods(llm_metrics: Dict,
                       bm25_metrics_file: str,
                       finetuned_metrics_file: str):
    """
    对比三种方法的性能

    Args:
        llm_metrics: LLM 重排指标
        bm25_metrics_file: BM25 指标文件
        finetuned_metrics_file: 微调模型指标文件
    """
    # 加载其他方法的指标
    with open(bm25_metrics_file, 'r', encoding='utf-8') as f:
        bm25_metrics = json.load(f)

    with open(finetuned_metrics_file, 'r', encoding='utf-8') as f:
        finetuned_metrics = json.load(f)

    print("\n" + "="*80)
    print("三种方法性能对比")
    print("="*80)

    # 对比关键指标
    key_metrics = [
        ('NDCG', 'NDCG@10'),
        ('MAP', 'MAP@10'),
        ('Recall', 'Recall@10'),
        ('Precision', 'P@10'),
        ('MRR', 'MRR@10')
    ]

    print(f"\n{'指标':<15} {'BM25':<12} {'微调模型':<12} {'LLM重排':<12} {'最佳方法':<12}")
    print("-" * 80)

    for metric_type, metric_key in key_metrics:
        bm25_value = bm25_metrics[metric_type][metric_key]
        finetuned_value = finetuned_metrics[metric_type][metric_key]
        llm_value = llm_metrics[metric_type][metric_key]

        # 找出最佳方法
        values = {
            'BM25': bm25_value,
            '微调模型': finetuned_value,
            'LLM重排': llm_value
        }
        best_method = max(values, key=values.get)

        print(f"{metric_key:<15} {bm25_value:<12.4f} {finetuned_value:<12.4f} {llm_value:<12.4f} {best_method:<12}")

    # 计算提升百分比
    print("\n" + "="*80)
    print("相对 BM25 的提升百分比")
    print("="*80)
    print(f"\n{'指标':<15} {'微调模型':<20} {'LLM重排':<20}")
    print("-" * 80)

    for metric_type, metric_key in key_metrics:
        bm25_value = bm25_metrics[metric_type][metric_key]
        finetuned_value = finetuned_metrics[metric_type][metric_key]
        llm_value = llm_metrics[metric_type][metric_key]

        finetuned_improvement = ((finetuned_value - bm25_value) / bm25_value) * 100
        llm_improvement = ((llm_value - bm25_value) / bm25_value) * 100

        print(f"{metric_key:<15} {finetuned_improvement:+.2f}%{' '*13} {llm_improvement:+.2f}%")


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


def main():
    """
    主函数：使用阿里云百炼 API 进行重排实验（改进的 Listwise 方法）
    """
    print("="*60)
    print("大语言模型重排实验（改进的 Listwise 方法）")
    print("="*60)

    # 配置参数
    TOP_K = 10  # 重排前 10 个文档
    MODEL = "qwen-max"  # 使用 qwen-max（最强模型）
    MAX_QUERIES = None  # None 表示处理所有查询，可以设置为 50 进行测试

    print(f"\n改进说明:")
    print(f"  1. 使用中文 Prompt（更适合 Qwen 模型）")
    print(f"  2. 添加 Few-shot 示例（教模型如何输出）")
    print(f"  3. 更强的格式约束（逗号分隔，无其他文字）")
    print(f"  4. 多重解析策略（支持多种输出格式）")
    print(f"  5. 极低温度（temperature=0.01，更确定的输出）")
    print(f"  6. 升级到 qwen-max（最强理解和遵循指令能力）")

    print(f"\n配置参数:")
    print(f"  - 重排文档数: {TOP_K}")
    print(f"  - 使用模型: {MODEL} (最强性能)")
    print(f"  - 预计耗时: 5-10 分钟")
    print(f"  - 预计成本: 2-4 元人民币 (qwen-max 价格较高)")

    # 获取 API Key
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("\n错误: 未找到 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量:")
        print("  Windows PowerShell: $env:DASHSCOPE_API_KEY='your_api_key'")
        print("  Windows CMD: set DASHSCOPE_API_KEY=your_api_key")
        print("  Linux/Mac: export DASHSCOPE_API_KEY=your_api_key")
        print("\n或者在代码中直接设置:")
        api_key = input("请输入你的阿里云百炼 API Key: ").strip()
        if not api_key:
            print("未提供 API Key，退出程序")
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

    # 3. LLM 重排
    print("\n步骤 3: 使用 LLM API 进行重排（改进的 Listwise 方法）")
    reranked_results = llm_rerank_all(
        api_key, queries, corpus, bm25_results,
        top_k=TOP_K, model=MODEL, max_queries=MAX_QUERIES
    )

    # 4. 评估结果
    print("\n步骤 4: 评估重排性能")
    metrics = evaluate_results(reranked_results, qrels)

    # 5. 打印结果
    print_metrics(metrics, f"LLM Re-ranking Improved ({MODEL})")

    # 6. 三方对比
    print("\n步骤 5: 三方对比分析")
    compare_all_methods(
        metrics,
        "results/bm25_metrics.json",
        "results/finetuned_biencoder_metrics.json"
    )

    # 7. 保存结果
    print("\n步骤 6: 保存结果")
    save_results(reranked_results, metrics, "llm_reranking_improved")

    print("\n" + "="*60)
    print("LLM 重排实验完成！")
    print("="*60)
    print("\n主要改进:")
    print("  1. 中文 Prompt + Few-shot 示例")
    print("  2. 简化输出格式（逗号分隔）")
    print("  3. 多重解析策略（容错性更强）")
    print("  4. 极低温度（更确定的输出）")


if __name__ == "__main__":
    main()
