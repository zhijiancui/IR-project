"""
准备训练数据
从NFCorpus数据集中构造训练样本对 (query, positive_document)
用于对比学习微调
"""
import os
import json
import pathlib
from typing import Dict, List, Tuple
from beir.datasets.data_loader import GenericDataLoader


def prepare_training_examples(corpus: Dict, queries: Dict, qrels: Dict) -> List[Tuple[str, str]]:
    """
    准备训练样本对

    Args:
        corpus: 文档库 {doc_id: {'title': ..., 'text': ...}}
        queries: 查询 {query_id: query_text}
        qrels: 相关性标注 {query_id: {doc_id: relevance_score}}

    Returns:
        training_examples: [(query_text, positive_doc_text), ...]
    """
    print("正在构造训练样本...")
    training_examples = []

    for query_id, query_text in queries.items():
        # 获取该查询的相关文档
        if query_id not in qrels:
            continue

        relevant_docs = qrels[query_id]

        # 对于每个相关文档，创建一个训练样本
        for doc_id, relevance_score in relevant_docs.items():
            # 只使用相关度 >= 1 的文档作为正样本
            if relevance_score >= 1:
                if doc_id in corpus:
                    # 合并文档标题和正文
                    doc_title = corpus[doc_id].get('title', '')
                    doc_text = corpus[doc_id].get('text', '')
                    full_doc_text = f"{doc_title} {doc_text}".strip()

                    # 添加训练样本
                    training_examples.append((query_text, full_doc_text))

    print(f"构造完成！共 {len(training_examples)} 个训练样本")
    return training_examples


def split_train_dev(examples: List[Tuple[str, str]],
                    dev_ratio: float = 0.1) -> Tuple[List, List]:
    """
    划分训练集和验证集

    Args:
        examples: 训练样本列表
        dev_ratio: 验证集比例

    Returns:
        train_examples: 训练集
        dev_examples: 验证集
    """
    import random
    random.seed(42)

    # 打乱数据
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # 划分
    split_idx = int(len(shuffled) * (1 - dev_ratio))
    train_examples = shuffled[:split_idx]
    dev_examples = shuffled[split_idx:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_examples)} 样本")
    print(f"  验证集: {len(dev_examples)} 样本")

    return train_examples, dev_examples


def save_training_data(train_examples: List, dev_examples: List,
                       output_dir: str = "data"):
    """
    保存训练数据

    Args:
        train_examples: 训练集
        dev_examples: 验证集
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练集
    train_file = os.path.join(output_dir, "train_pairs.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    print(f"\n训练集已保存到: {train_file}")

    # 保存验证集
    dev_file = os.path.join(output_dir, "dev_pairs.json")
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_examples, f, indent=2, ensure_ascii=False)
    print(f"验证集已保存到: {dev_file}")


def analyze_data_statistics(train_examples: List, dev_examples: List):
    """
    分析数据统计信息

    Args:
        train_examples: 训练集
        dev_examples: 验证集
    """
    print("\n" + "="*60)
    print("数据统计分析")
    print("="*60)

    # 计算平均长度
    train_query_lens = [len(ex[0].split()) for ex in train_examples]
    train_doc_lens = [len(ex[1].split()) for ex in train_examples]

    print(f"\n训练集统计:")
    print(f"  样本数量: {len(train_examples)}")
    print(f"  查询平均长度: {sum(train_query_lens)/len(train_query_lens):.1f} 词")
    print(f"  文档平均长度: {sum(train_doc_lens)/len(train_doc_lens):.1f} 词")
    print(f"  查询最大长度: {max(train_query_lens)} 词")
    print(f"  文档最大长度: {max(train_doc_lens)} 词")

    dev_query_lens = [len(ex[0].split()) for ex in dev_examples]
    dev_doc_lens = [len(ex[1].split()) for ex in dev_examples]

    print(f"\n验证集统计:")
    print(f"  样本数量: {len(dev_examples)}")
    print(f"  查询平均长度: {sum(dev_query_lens)/len(dev_query_lens):.1f} 词")
    print(f"  文档平均长度: {sum(dev_doc_lens)/len(dev_doc_lens):.1f} 词")

    # 显示一些示例
    print("\n" + "="*60)
    print("训练样本示例")
    print("="*60)

    for i in range(min(3, len(train_examples))):
        query, doc = train_examples[i]
        print(f"\n示例 {i+1}:")
        print(f"查询: {query}")
        print(f"文档: {doc[:200]}...")


def main():
    """
    主函数：准备训练数据
    """
    print("="*60)
    print("准备对比学习训练数据")
    print("="*60)

    # 1. 加载训练集数据
    print("\n步骤 1: 加载NFCorpus训练集")
    data_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                             "datasets", "nfcorpus")

    # 加载训练集
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    print(f"文档数量: {len(corpus)}")
    print(f"查询数量: {len(queries)}")
    print(f"相关性标注数量: {len(qrels)}")

    # 2. 构造训练样本
    print("\n步骤 2: 构造训练样本对")
    training_examples = prepare_training_examples(corpus, queries, qrels)

    # 3. 划分训练集和验证集
    print("\n步骤 3: 划分训练集和验证集")
    train_examples, dev_examples = split_train_dev(training_examples, dev_ratio=0.1)

    # 4. 保存数据
    print("\n步骤 4: 保存训练数据")
    save_training_data(train_examples, dev_examples)

    # 5. 数据统计分析
    analyze_data_statistics(train_examples, dev_examples)

    print("\n" + "="*60)
    print("训练数据准备完成！")
    print("="*60)


if __name__ == "__main__":
    main()
