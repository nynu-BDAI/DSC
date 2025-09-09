import json

# 1. 读取 json 文件
with open('/mnt/Data/wangshilong/self_datasets/miniimagenet/train_wind_to_label.json', 'r') as f:
    id2wnid = json.load(f)

# 注意：JSON 读进来的是字符串键，转换为整数键
id2wnid = {int(k): v for k, v in id2wnid.items()} #{0: 'n01440764', 1: 'n01443537', ...}


# 2. 读取 txt 文件，构造 wnid 到 label 的映射
wnid2label = {}
with open('/mnt/Data/wangshilong/self_datasets/miniimagenet/synset_words.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            wnid, label = parts
            wnid2label[wnid] = label

# 3. 构造 id 到文本标签的映射
id2textlabel = {k: wnid2label.get(v, "UNKNOWN") for k, v in id2wnid.items()} #k:0,1,2...; v:n01440764,n01443537...

# ✅ 可选：保存到新的 JSON 文件
with open('/mnt/Data/wangshilong/self_datasets/miniimagenet/id2textlabel.json', 'w') as f:
    json.dump(id2textlabel, f, indent=2)

# ✅ 打印查看
print(id2textlabel)