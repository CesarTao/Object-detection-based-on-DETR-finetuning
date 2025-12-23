import os
# 1. 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. (可选) 强制指定缓存到数据盘，防止系统盘爆满
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
import os
from datasets import load_dataset
from PIL import Image, ImageDraw
from io import BytesIO
import collections

# ================= 配置 =================
LOCAL_DATA_DIR = "/root/autodl-tmp/coco_full"
# =======================================

print("🚀 正在加载数据进行“身世调查”...")
data_files = {"train": os.path.join(LOCAL_DATA_DIR, "data/train-*.parquet")}

# 我们加载前 100 张图，足够覆盖大部分类别了
dataset = load_dataset("parquet", data_files=data_files, split="train[:100]")

# ----------------- 1. 统计 ID 分布 -----------------
print("\n📊 正在统计 ID 分布...")
all_ids = set()
id_counts = collections.Counter()

for item in dataset:
    objects = item['objects']
    cats = objects['category']
    for c in cats:
        # 你的 category 可能是 int 也可能是 float，统一转 int 看
        c = int(c)
        all_ids.add(c)
        id_counts[c] += 1

sorted_ids = sorted(list(all_ids))
print(f"✅ 统计完成！")
print(f"最小 ID: {min(sorted_ids) if sorted_ids else '无'}")
print(f"最大 ID: {max(sorted_ids) if sorted_ids else '无'}")
print(f"总共有 {len(sorted_ids)} 种不同的 ID")
print(f"ID 列表 (前 20 个): {sorted_ids[:20]} ...")
print(f"ID 列表 (后 10 个): ... {sorted_ids[-10:]}")

# ----------------- 2. 这里的判断逻辑 -----------------
print("\n🧐 自动分析结果：")
if 0 in sorted_ids:
    print("👉 发现 ID 0：这通常意味着数据已经是【0-indexed】(0-79) 或者包含了背景类。")
else:
    print("👉 没有 ID 0：最小是 1。")

if 90 in sorted_ids:
    print("👉 发现 ID 90：这是铁证！你的数据是【标准 COCO 格式】(1-90)。")
    print("💡 结论：你绝对需要上面的 Mapping 代码 (把 90 映射回 79)。")
elif max(sorted_ids) <= 79:
    print("👉 最大 ID <= 79：这看起来像是已经映射过的数据 (0-79)。")
else:
    print("👉 ID 情况比较奇怪，请把上面的 ID 列表发给我分析。")

# ----------------- 3. 视觉验证 (眼见为实) -----------------
print("\n🖼️ 正在生成一张可视化的“证据图”...")
# 找一张物体比较多的图
target_idx = 0
max_objs = 0
for i, item in enumerate(dataset):
    if len(item['objects']['category']) > max_objs:
        max_objs = len(item['objects']['category'])
        target_idx = i

# 取出这张图
sample = dataset[target_idx]
img_data = sample['image']
if isinstance(img_data, dict) and 'bytes' in img_data:
    image = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
else:
    image = img_data.convert("RGB")

draw = ImageDraw.Draw(image)
objects = sample['objects']

print(f"选取了第 {target_idx} 张图，包含 {len(objects['category'])} 个物体。")

for box, cat in zip(objects['bbox'], objects['category']):
    # 你的 BBox 逻辑是 [xmin, ymin, xmax, ymax]
    x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    
    # 画框
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    
    # 🚨 重点：只写 ID，不写名字 (因为现在还不知道名字对不对)
    # 我们把 ID 写大一点
    text = f"ID: {cat}"
    draw.text((x_min, y_min), text, fill='red') # 也可以加个背景色看不清楚的话

image.save("check_id_truth.jpg")
print(f"✅ 图片已保存为 'check_id_truth.jpg'。")
print("👉 请打开图片，看着红框里的物体：")
print("   - 如果框住的是【人】，且 ID 写着 【1】：说明数据是 COCO 标准格式 (需映射)。")
print("   - 如果框住的是【人】，且 ID 写着 【0】：说明数据是 0-索引格式 (无需复杂映射)。")