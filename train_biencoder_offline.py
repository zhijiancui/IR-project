"""
Bi-Encoder微调脚本（离线模式）
使用对比学习在NFCorpus数据上微调预训练模型
从本地缓存加载模型，无需网络连接
"""
import os
import json
import time
from datetime import datetime
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def load_training_data(train_file: str, dev_file: str) -> Tuple[List, List]:
    """加载训练数据"""
    print("正在加载训练数据...")

    with open(train_file, 'r', encoding='utf-8') as f:
        train_pairs = json.load(f)

    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_pairs = json.load(f)

    print(f"训练集: {len(train_pairs)} 样本")
    print(f"验证集: {len(dev_pairs)} 样本")

    return train_pairs, dev_pairs


def create_input_examples(pairs: List[Tuple[str, str]]) -> List[InputExample]:
    """创建InputExample对象"""
    examples = []
    for query, doc in pairs:
        examples.append(InputExample(texts=[query, doc]))
    return examples


def train_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                train_file: str = 'data/train_pairs.json',
                dev_file: str = 'data/dev_pairs.json',
                output_dir: str = 'models/finetuned-medical-retriever',
                batch_size: int = 64,
                epochs: int = 3,
                warmup_steps: int = 500):
    """训练Bi-Encoder模型（离线模式）"""

    print("="*60)
    print("Bi-Encoder 对比学习微调（离线模式）")
    print("="*60)
    print(f"模式: 离线（从本地缓存加载）")

    # 1. 加载预训练模型（从本地缓存）
    print(f"\n步骤 1: 从本地缓存加载预训练模型 '{model_name}'")
    print("正在加载模型...")

    try:
        # 强制禁用CUDA
        import torch
        torch.cuda.is_available = lambda: False

        # 从本地缓存加载模型并指定CPU设备
        model = SentenceTransformer(model_name, device='cpu')
        print(f"模型加载成功！")
        print(f"模型维度: {model.get_sentence_embedding_dimension()}")
        print(f"运行设备: CPU")

    except Exception as e:
        print(f"\n错误: 无法加载模型")
        print(f"错误信息: {str(e)}")
        raise

    # 2. 加载训练数据
    print(f"\n步骤 2: 加载训练数据")
    train_pairs, dev_pairs = load_training_data(train_file, dev_file)

    # 3. 创建InputExample
    print(f"\n步骤 3: 准备训练样本")
    train_examples = create_input_examples(train_pairs)
    print(f"训练样本准备完成: {len(train_examples)} 个")

    # 4. 创建DataLoader
    print(f"\n步骤 4: 创建DataLoader (batch_size={batch_size})")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    print(f"每个epoch的batch数: {len(train_dataloader)}")
    print(f"总训练步数: {len(train_dataloader) * epochs}")

    # 5. 定义损失函数
    print(f"\n步骤 5: 定义损失函数")
    print("使用 MultipleNegativesRankingLoss (对比学习)")
    print("原理: Batch内的其他正样本作为负样本")
    print(f"当前batch_size={batch_size}, 意味着每个样本有{batch_size-1}个负样本")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 6. 配置训练参数
    print(f"\n步骤 6: 配置训练参数")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - Batch大小: {batch_size}")
    print(f"  - 预热步数: {warmup_steps}")
    print(f"  - 学习率: 2e-5 (默认)")

    # 7. 开始训练
    print(f"\n步骤 7: 开始训练")
    print("="*60)

    start_time = time.time()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True,
        checkpoint_save_steps=1000,
        checkpoint_path=output_dir + "/checkpoints",
    )

    elapsed_time = time.time() - start_time

    print("="*60)
    print(f"训练完成！总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"模型已保存到: {output_dir}")

    # 8. 保存训练信息
    training_info = {
        "model_name": model_name,
        "train_samples": len(train_examples),
        "dev_samples": len(dev_pairs),
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "training_time_minutes": elapsed_time / 60,
        "device": device,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "offline"
    }

    os.makedirs(output_dir, exist_ok=True)
    info_file = os.path.join(output_dir, "training_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"训练信息已保存到: {info_file}")

    return model


def main():
    """主函数：训练Bi-Encoder"""

    # 训练配置
    config = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'train_file': 'data/train_pairs.json',
        'dev_file': 'data/dev_pairs.json',
        'output_dir': 'models/finetuned-medical-retriever',
        'batch_size': 64,  # RTX 5070 Ti可以用大batch
        'epochs': 3,
        'warmup_steps': 500
    }

    print("\n" + "="*60)
    print("训练配置（离线模式）")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("提示:")
    print("  - 离线模式：从本地缓存加载模型")
    print("  - 无需网络连接")
    print("  - 使用GPU加速训练")
    print("  - 可以随时按Ctrl+C中断训练")
    print()

    # 开始训练
    try:
        model = train_model(**config)

        print("\n" + "="*60)
        print("Bi-Encoder微调完成！")
        print("="*60)
        print("\n下一步:")
        print("  1. 运行 dense_retrieval.py 使用微调后的模型进行检索")
        print("  2. 对比BM25和微调模型的性能差异")
        print("  3. 可视化结果对比")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print("部分训练的模型可能已保存在检查点目录")
    except Exception as e:
        print(f"\n\n训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
