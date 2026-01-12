"""
数据加载脚本 - 下载并加载NFCorpus医疗检索数据集
"""
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib
import os

def load_nfcorpus_dataset():
    """
    下载并加载NFCorpus数据集

    Returns:
        corpus: 文档库 (dict)
        queries: 查询问题 (dict)
        qrels: 标准答案/相关性标注 (dict)
    """
    # 数据集名称
    dataset = "nfcorpus"

    # 数据集下载URL
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    # 设置数据存储路径
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

    print(f"正在下载 {dataset} 数据集...")
    print(f"下载URL: {url}")
    print(f"保存路径: {out_dir}")

    # 下载并解压数据集
    data_path = util.download_and_unzip(url, out_dir)

    print(f"数据集已下载到: {data_path}")

    # 加载数据集
    print("正在加载数据集...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # 打印数据集统计信息
    print("\n" + "="*50)
    print("数据集加载完成！")
    print("="*50)
    print(f"文档数量 (corpus): {len(corpus)}")
    print(f"查询数量 (queries): {len(queries)}")
    print(f"相关性标注数量 (qrels): {len(qrels)}")

    # 显示示例数据
    print("\n" + "="*50)
    print("示例查询:")
    print("="*50)
    first_query_id = list(queries.keys())[0]
    print(f"Query ID: {first_query_id}")
    print(f"Query Text: {queries[first_query_id]}")

    print("\n" + "="*50)
    print("示例文档:")
    print("="*50)
    first_doc_id = list(corpus.keys())[0]
    print(f"Doc ID: {first_doc_id}")
    print(f"Title: {corpus[first_doc_id].get('title', 'N/A')}")
    print(f"Text (前200字符): {corpus[first_doc_id]['text'][:200]}...")

    return corpus, queries, qrels

if __name__ == "__main__":
    # 运行数据加载
    corpus, queries, qrels = load_nfcorpus_dataset()

    print("\n" + "="*50)
    print("数据加载脚本执行完成！")
    print("="*50)
    print("你现在可以使用这些数据进行信息检索实验了。")
