"""
可视化检索结果
绘制各种评估指标的对比图表
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_metrics(metrics_file: str) -> dict:
    """
    加载评估指标

    Args:
        metrics_file: 指标文件路径

    Returns:
        metrics: 指标字典
    """
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


def plot_metrics_comparison(metrics_dict: dict, output_dir: str = "results"):
    """
    绘制不同模型的指标对比图

    Args:
        metrics_dict: {model_name: metrics} 字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. NDCG对比图
    plt.figure(figsize=(12, 6))

    x_labels = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@100']
    x_pos = np.arange(len(x_labels))
    width = 0.35

    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        ndcg_values = [
            metrics['NDCG']['NDCG@1'],
            metrics['NDCG']['NDCG@3'],
            metrics['NDCG']['NDCG@5'],
            metrics['NDCG']['NDCG@10'],
            metrics['NDCG']['NDCG@100']
        ]
        plt.bar(x_pos + idx * width, ndcg_values, width, label=model_name, alpha=0.8)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('NDCG Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos + width / 2, x_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'ndcg_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"NDCG对比图已保存到: {output_file}")
    plt.close()

    # 2. Recall对比图
    plt.figure(figsize=(12, 6))

    x_labels = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@100']

    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        recall_values = [
            metrics['Recall']['Recall@1'],
            metrics['Recall']['Recall@3'],
            metrics['Recall']['Recall@5'],
            metrics['Recall']['Recall@10'],
            metrics['Recall']['Recall@100']
        ]
        plt.bar(x_pos + idx * width, recall_values, width, label=model_name, alpha=0.8)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Recall Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos + width / 2, x_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Recall对比图已保存到: {output_file}")
    plt.close()

    # 3. MAP对比图
    plt.figure(figsize=(12, 6))

    x_labels = ['MAP@1', 'MAP@3', 'MAP@5', 'MAP@10', 'MAP@100']

    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        map_values = [
            metrics['MAP']['MAP@1'],
            metrics['MAP']['MAP@3'],
            metrics['MAP']['MAP@5'],
            metrics['MAP']['MAP@10'],
            metrics['MAP']['MAP@100']
        ]
        plt.bar(x_pos + idx * width, map_values, width, label=model_name, alpha=0.8)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('MAP Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos + width / 2, x_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'map_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"MAP对比图已保存到: {output_file}")
    plt.close()

    # 4. 综合指标对比（雷达图）
    plot_radar_chart(metrics_dict, output_dir)


def plot_radar_chart(metrics_dict: dict, output_dir: str):
    """
    绘制雷达图展示综合性能

    Args:
        metrics_dict: {model_name: metrics} 字典
        output_dir: 输出目录
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # 选择关键指标
    categories = ['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'MRR@10']
    num_vars = len(categories)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    for model_name, metrics in metrics_dict.items():
        values = [
            metrics['NDCG']['NDCG@10'],
            metrics['MAP']['MAP@10'],
            metrics['Recall']['Recall@10'],
            metrics['Precision']['P@10'],
            metrics['MRR']['MRR@10']
        ]
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.set_title('Overall Performance Comparison (Top-10)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    output_file = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存到: {output_file}")
    plt.close()


def plot_single_model_summary(model_name: str, metrics: dict, output_dir: str = "results"):
    """
    为单个模型绘制性能总结图

    Args:
        model_name: 模型名称
        metrics: 评估指标
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Performance Summary', fontsize=16, fontweight='bold')

    # 1. NDCG
    ax = axes[0, 0]
    ndcg_keys = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@100']
    ndcg_values = [metrics['NDCG'][k] for k in ndcg_keys]
    bars = ax.bar(range(len(ndcg_keys)), ndcg_values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(ndcg_keys)))
    ax.set_xticklabels(ndcg_keys, rotation=45)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('NDCG Scores', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 2. Recall
    ax = axes[0, 1]
    recall_keys = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@100']
    recall_values = [metrics['Recall'][k] for k in recall_keys]
    bars = ax.bar(range(len(recall_keys)), recall_values, color='forestgreen', alpha=0.8)
    ax.set_xticks(range(len(recall_keys)))
    ax.set_xticklabels(recall_keys, rotation=45)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Recall Scores', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. MAP
    ax = axes[1, 0]
    map_keys = ['MAP@1', 'MAP@3', 'MAP@5', 'MAP@10', 'MAP@100']
    map_values = [metrics['MAP'][k] for k in map_keys]
    bars = ax.bar(range(len(map_keys)), map_values, color='coral', alpha=0.8)
    ax.set_xticks(range(len(map_keys)))
    ax.set_xticklabels(map_keys, rotation=45)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('MAP Scores', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 4. Precision
    ax = axes[1, 1]
    precision_keys = ['P@1', 'P@3', 'P@5', 'P@10', 'P@100']
    precision_values = [metrics['Precision'][k] for k in precision_keys]
    bars = ax.bar(range(len(precision_keys)), precision_values, color='mediumpurple', alpha=0.8)
    ax.set_xticks(range(len(precision_keys)))
    ax.set_xticklabels(precision_keys, rotation=45)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Precision Scores', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_file = os.path.join(output_dir, f'{model_name.lower()}_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{model_name}性能总结图已保存到: {output_file}")
    plt.close()


def main():
    """
    主函数：可视化检索结果（BM25 vs 微调模型）
    """
    print("="*60)
    print("检索结果可视化")
    print("="*60)

    # 加载BM25指标
    bm25_metrics_file = "results/bm25_metrics.json"
    finetuned_metrics_file = "results/finetuned_biencoder_metrics.json"

    if not os.path.exists(bm25_metrics_file):
        print(f"错误: 找不到指标文件 {bm25_metrics_file}")
        print("请先运行 bm25_baseline.py 生成评估指标")
        return

    print(f"\n正在加载BM25指标: {bm25_metrics_file}")
    bm25_metrics = load_metrics(bm25_metrics_file)

    # 为BM25绘制单独的性能总结图
    print("\n正在生成BM25性能总结图...")
    plot_single_model_summary("BM25", bm25_metrics)

    # 创建对比字典
    metrics_dict = {"BM25": bm25_metrics}

    # 如果微调模型指标存在，添加到对比中
    if os.path.exists(finetuned_metrics_file):
        print(f"\n正在加载微调模型指标: {finetuned_metrics_file}")
        finetuned_metrics = load_metrics(finetuned_metrics_file)

        print("\n正在生成微调模型性能总结图...")
        plot_single_model_summary("Finetuned Bi-Encoder", finetuned_metrics)

        # 添加到对比字典
        metrics_dict["Finetuned Bi-Encoder"] = finetuned_metrics

        print("\n正在生成对比图...")
        plot_metrics_comparison(metrics_dict)

        # 打印性能对比
        print("\n" + "="*60)
        print("性能对比总结")
        print("="*60)
        print(f"\n{'指标':<15} {'BM25':<12} {'微调模型':<12} {'提升':<12}")
        print("-" * 60)

        key_metrics = [
            ('NDCG', 'NDCG@10'),
            ('MAP', 'MAP@10'),
            ('Recall', 'Recall@10'),
            ('Precision', 'P@10'),
            ('MRR', 'MRR@10')
        ]

        for metric_type, metric_key in key_metrics:
            bm25_value = bm25_metrics[metric_type][metric_key]
            finetuned_value = finetuned_metrics[metric_type][metric_key]
            improvement = ((finetuned_value - bm25_value) / bm25_value) * 100
            print(f"{metric_key:<15} {bm25_value:<12.4f} {finetuned_value:<12.4f} {improvement:+.2f}%")
    else:
        print(f"\n提示: 未找到微调模型指标文件 {finetuned_metrics_file}")
        print("只生成BM25的可视化结果")
        print("\n正在生成对比图...")
        plot_metrics_comparison(metrics_dict)

    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)
    print("\n生成的图表:")
    print("  - results/bm25_summary.png (BM25性能总结)")
    if os.path.exists(finetuned_metrics_file):
        print("  - results/finetuned bi-encoder_summary.png (微调模型性能总结)")
    print("  - results/ndcg_comparison.png (NDCG对比)")
    print("  - results/recall_comparison.png (Recall对比)")
    print("  - results/map_comparison.png (MAP对比)")
    print("  - results/radar_comparison.png (雷达图)")


if __name__ == "__main__":
    main()
