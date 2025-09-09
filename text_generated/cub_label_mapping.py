def process_class_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # 按空格拆分成两个部分：id 和 name
            idx, name = line.split(" ", 1)

            # 减 1 作为新索引
            new_idx = int(idx) - 1

            # 去除前缀编号（例如 "001."）
            if "." in name:
                name = name.split(".", 1)[1]

            # 写入新的一行
            fout.write(f"{new_idx} {name}\n")

    print(f"✅ 已保存到：{output_path}")

if __name__ == "__main__":
    input_file = "/mnt/Data/wangshilong/self_datasets/CUB_200_2011/classes.txt"
    output_file = "/mnt/Data/wangshilong/self_datasets/CUB_200_2011/classes_mapping.txt"
    process_class_file(input_file, output_file)
    