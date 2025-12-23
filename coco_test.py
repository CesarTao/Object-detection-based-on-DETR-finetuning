import os
from datasets import load_dataset

# 指向你的本地数据文件夹
local_dataset_path = "/root/autodl-tmp/coco_full"

# --- 核心修改 ---
# 指定一个在数据盘的缓存路径
cache_dir = "/root/autodl-tmp/hf_cache_work"
os.makedirs(cache_dir, exist_ok=True)

print(f"正在加载数据，缓存将写入: {cache_dir}")

dataset = load_dataset(
    local_dataset_path, 
    split="val[:10]", 
    cache_dir=cache_dir  # <--- 加上这一句，把压力转移到数据盘
)


print("本地加载成功！")

# 2. 查看数据集的整体结构
print(f"\n数据集大小: {len(dataset)} 张图片")
print(f"数据特征 (Features): {dataset.features}")

# 3. 取出第一条数据看看长什么样
sample = dataset[0]
print("\n--- 第一条样本详情 ---")
print(f"Keys: {sample.keys()}")
print(f"图片对象: {sample['image']}") # 应该是 PIL.JpegImagePlugin.JpegImageFile
print(f"图片尺寸: {sample['image'].size}") # (Width, Height)
print(f"标注信息 (Objects): {sample['objects']}")