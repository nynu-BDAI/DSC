import csv
import pandas as pd
from pathlib import Path
import argparse

def load_synset_words(path: Path):
    """
    读取 synset_words.txt，返回 {synset_id: "names"} 的 dict
    行格式示例： n01440764 tench, Tinca tinca
    注意：只在第一个空格处分割，后面的逗号原样保留
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                sid, names = parts[0].strip(), parts[1].strip()
            else:
                sid, names = parts[0].strip(), ""
            mapping[sid] = names
    return mapping

def main(train_csv: Path, synset_words_txt: Path, out_csv: Path):
    # 1) 读 synset 映射
    mapping = load_synset_words(synset_words_txt)

    # 2) 读 train.csv（假设前两列依次为：文件名、标签）
    df = pd.read_csv(train_csv)

    df["synset_words"] = df["label"].astype(str).map(mapping)

    # 4) 导出
    df.to_csv(out_csv, index=False)
    print(f"✅ 已生成：{out_csv}")


if __name__ == "__main__":
    # main(Path("/mnt/Data/wangshilong/self_datasets/miniimagenet/split/train.csv"),
    #      Path("/mnt/Data/wangshilong/self_datasets/miniimagenet/synset_words.txt"),
    #      Path("/mnt/Data/wangshilong/self_datasets/miniimagenet/train_with_synset.csv"))

    csv_path = "/mnt/Data/wangshilong/self_datasets/miniimagenet/train_with_synset.csv"
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 读取表头
        header = next(reader)
        print("列名：", header) #列名： ['filename', 'label']
        # 遍历数据
        for row in reader:
            print(row)





    