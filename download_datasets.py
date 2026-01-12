"""
批量下载BEIR数据集
下载多个标准数据集用于多数据集评估
"""
import os
import sys
from pathlib import Path
from beir import util
from beir.datasets.data_loader import GenericDataLoader


# 要下载的数据集列表
DATASETS = [
    {
        'name': 'nfcorpus',
        'description': '医学领域（小规模）',
        'corpus_size': '3,633',
        'queries': '323'
    },
    {
        'name': 'scifact',
        'description': '科学事实验证（小规模）',
        'corpus_size': '5,183',
        'queries': '300'
    },
    {
        'name': 'fiqa',
        'description': '金融问答（中等规模）',
        'corpus_size': '57,638',
        'queries': '648'
    },
    {
        'name': 'trec-covid',
        'description': 'COVID-19文献（中等规模）',
        'corpus_size': '171,332',
        'queries': '50'
    },
    {
        'name': 'scidocs',
        'description': '科学文献引用（中等规模）',
        'corpus_size': '25,657',
        'queries': '1,000'
    },
]

# 数据集保存路径
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)


def check_dataset_exists(dataset_name: str) -> bool:
    """检查数据集是否已存在"""
    dataset_path = DATASETS_DIR / dataset_name

    if not dataset_path.exists():
        return False

    # 检查必要的文件是否存在
    required_files = ['corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv']
    for file in required_files:
        if not (dataset_path / file).exists():
            return False

    return True


def download_dataset(dataset_name: str, description: str) -> bool:
    """下载单个数据集"""
    print(f"\n{'='*60}")
    print(f"下载数据集: {dataset_name}")
    print(f"描述: {description}")
    print(f"{'='*60}")

    # 检查是否已存在
    if check_dataset_exists(dataset_name):
        print(f"[OK] 数据集已存在，跳过下载")
        return True

    try:
        # 下载数据集
        print(f"正在下载 {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, str(DATASETS_DIR))

        print(f"[OK] 下载完成: {data_path}")

        # 验证数据集
        print(f"验证数据集...")
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        print(f"[OK] 验证成功:")
        print(f"  - 语料库大小: {len(corpus):,}")
        print(f"  - 查询数量: {len(queries):,}")
        print(f"  - 相关判断数量: {sum(len(v) for v in qrels.values()):,}")

        return True

    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        return False


def verify_all_datasets():
    """验证所有数据集"""
    print(f"\n{'='*60}")
    print("验证所有数据集")
    print(f"{'='*60}\n")

    results = []

    for dataset in DATASETS:
        name = dataset['name']
        exists = check_dataset_exists(name)

        status = "[OK]" if exists else "[NO]"
        results.append({
            'name': name,
            'description': dataset['description'],
            'exists': exists,
            'status': status
        })

    # 打印验证结果表格
    print(f"{'数据集':<15} {'描述':<25} {'状态':<5}")
    print("-" * 50)

    for result in results:
        print(f"{result['name']:<15} {result['description']:<25} {result['status']:<5}")

    # 统计
    total = len(results)
    downloaded = sum(1 for r in results if r['exists'])

    print(f"\n总计: {downloaded}/{total} 个数据集已准备好")

    return downloaded == total


def get_dataset_statistics():
    """获取数据集统计信息"""
    print(f"\n{'='*60}")
    print("数据集统计信息")
    print(f"{'='*60}\n")

    print(f"{'数据集':<15} {'语料库':<12} {'查询':<8} {'相关判断':<10}")
    print("-" * 50)

    total_corpus = 0
    total_queries = 0
    total_qrels = 0

    for dataset in DATASETS:
        name = dataset['name']

        if not check_dataset_exists(name):
            print(f"{name:<15} {'未下载':<12} {'-':<8} {'-':<10}")
            continue

        try:
            data_path = str(DATASETS_DIR / name)
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

            corpus_size = len(corpus)
            query_count = len(queries)
            qrel_count = sum(len(v) for v in qrels.values())

            print(f"{name:<15} {corpus_size:<12,} {query_count:<8,} {qrel_count:<10,}")

            total_corpus += corpus_size
            total_queries += query_count
            total_qrels += qrel_count

        except Exception as e:
            print(f"{name:<15} {'错误':<12} {'-':<8} {'-':<10}")

    print("-" * 50)
    print(f"{'总计':<15} {total_corpus:<12,} {total_queries:<8,} {total_qrels:<10,}")


def main():
    """主函数"""
    # 检查是否使用自动模式
    auto_mode = '--auto' in sys.argv or '-y' in sys.argv

    print("="*60)
    print("BEIR数据集批量下载工具")
    print("="*60)

    # 显示将要下载的数据集
    print("\n将要下载以下数据集:\n")
    print(f"{'序号':<5} {'数据集':<15} {'描述':<25} {'语料库':<12} {'查询':<8}")
    print("-" * 70)

    for i, dataset in enumerate(DATASETS, 1):
        print(f"{i:<5} {dataset['name']:<15} {dataset['description']:<25} "
              f"{dataset['corpus_size']:<12} {dataset['queries']:<8}")

    print(f"\n总计: {len(DATASETS)} 个数据集")

    # 询问是否继续（除非使用自动模式）
    if not auto_mode:
        print("\n" + "="*60)
        response = input("是否开始下载? (y/n): ").strip().lower()

        if response != 'y':
            print("已取消下载")
            return
    else:
        print("\n自动模式：跳过确认，开始下载...")

    # 下载所有数据集
    print("\n开始下载数据集...")

    success_count = 0
    fail_count = 0

    for dataset in DATASETS:
        success = download_dataset(dataset['name'], dataset['description'])
        if success:
            success_count += 1
        else:
            fail_count += 1

    # 验证所有数据集
    all_ready = verify_all_datasets()

    # 显示统计信息
    if all_ready:
        get_dataset_statistics()

    # 总结
    print(f"\n{'='*60}")
    print("下载完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(DATASETS)}")
    print(f"失败: {fail_count}/{len(DATASETS)}")

    if all_ready:
        print("\n[OK] 所有数据集已准备就绪，可以运行评估了！")
        print("\n运行命令:")
        print("  python multi_dataset_evaluation.py")
    else:
        print("\n[FAIL] 部分数据集下载失败，请检查网络连接后重试")


if __name__ == "__main__":
    main()
