import os
import pandas as pd

# 1. 设置 Parquet 文件路径
PARQUET_PATH = "/root/autodl-tmp/coco_full/data/val-00000-of-00002-c4f2e391ee4aba11.parquet" 
# 注意：你需要去文件夹里看一眼具体的文件名，挑第一个就行，比如 train-00000-of-00xxx.parquet

# 2. 读取文件 (需要安装 pandas 和 pyarrow: pip install pandas pyarrow)
# 我们只读前 5 行，因为文件很大
df = pd.read_parquet(PARQUET_PATH).head(5)

# 3. 打印查看
print("列名:", df.columns)

# 查看第一张图的标注信息
first_row = df.iloc[0]
print("\n=== 第一张图片的标注信息 ===")
print(f"Image ID: {first_row.get('image_id', 'Unknown')}")

# 获取 objects 字段 (包含 bbox 和 category)
objects = first_row['objects'] # 这通常是一个字典，包含 'bbox' 和 'category' 两个列表
print("\n原始 Objects 数据:")
print(objects)

# 4. 解析一下让它更好看
print("\n=== 解析后的标签 ===")
bboxes = objects['bbox']
categories = objects['category']

for i, (box, cat_id) in enumerate(zip(bboxes, categories)):
    print(f"物体 {i+1}:")
    print(f"  - 类别 ID: {cat_id}")
    print(f"  - 坐标 Box: {box}")

import pandas as pd
from PIL import Image, ImageDraw
import io
import os


# 读取一行
df = pd.read_parquet(PARQUET_PATH).head(1)
row = df.iloc[0]

# 加载图片
image = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
draw = ImageDraw.Draw(image)

# 获取第一个框
box = row['objects']['bbox'][0]
print(f"原始 BBox数值: {box}")

# 🟥 假设 1：它是 COCO 标准格式 [x, y, w, h] (最可能)
# draw.rectangle 需要 [xmin, ymin, xmax, ymax]
# 所以如果是 xywh，我们需要转换：xmax = x + w, ymax = y + h
x, y, w, h = box
draw.rectangle([x, y, w, h], outline="red", width=5)
draw.text((x, y), "XYWH", fill="red")

# 🟦 假设 2：它是 [xmin, ymin, xmax, ymax]
# 如果它是这个格式，直接画就行
# draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=3)

# 保存查看
image.save("check_format.jpg")
print("已保存 check_format.jpg，请查看红框是否正确框住了物体。")