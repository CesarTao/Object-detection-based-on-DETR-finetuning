import os
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import os
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from io import BytesIO

# ================= 配置区域 =================
LOCAL_DATA_DIR = "/root/autodl-tmp/coco_full"
# 指向你刚才修复字典后保存的模型路径
MODEL_PATH = "/root/detr/detr-finetuned-coco-failed-backup" 
# ===========================================

print("🚀 正在初始化环境...")

# --- A. 加载本地 Hugging Face 数据集 ---
try:
    print(f"📂 正在从 {LOCAL_DATA_DIR} 加载数据...")
    data_files = {"validation": os.path.join(LOCAL_DATA_DIR, "data/val-*.parquet")}
    # 只加载前 5 张做测试
    dataset = load_dataset("parquet", data_files=data_files, split="validation[:5]")
    print(f"✅ 数据加载成功！共 {len(dataset)} 张图片。")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    # 备用方案
    dataset = load_dataset("detection-datasets/coco", split="validation[:5]", streaming=True)

# --- B. 加载 DETR 模型 ---
print(f"🧠 正在加载模型: {MODEL_PATH}")
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

# 💡 关键修改：直接从模型配置中获取字典，不再手动写列表
# 这样能确保和你训练/修复时的逻辑 100% 一致
id2label = model.config.id2label
print(f"✅ 成功加载类别映射，共 {len(id2label)} 类")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() # 开启评估模式
print(f"✅ 模型就绪！运行设备: {device}")


# --- C. 核心函数: 推理 + NMS ---
def detect(pil_img):
    # 1. 预处理
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. 后处理 (获取原始框)
    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    # threshold=0.5: 初筛阈值
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # ================= 🚀 NMS 过滤逻辑 =================
    boxes = results["boxes"]   # [x_min, y_min, x_max, y_max]
    scores = results["scores"]
    labels = results["labels"]

    # 如果没有检测到任何物体，直接返回空
    if boxes.shape[0] == 0:
        return []

    # iou_threshold=0.3: 去重力度。越小去重越狠（适合家具），越大越保留（适合密集物体）
    keep_indices = nms(boxes, scores, iou_threshold=0.3)

    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]
    # ===================================================

    detections = []
    for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
        score = score.item()
        label_id = label.item()
        
        # 💡 关键修改：健壮的字典查找
        # JSON 里的 key 可能是字符串 "1"，也可能是整数 1，这里做个兼容
        if label_id in id2label:
            label_name = id2label[label_id]
        elif str(label_id) in id2label:
            label_name = id2label[str(label_id)]
        else:
            label_name = f"Unknown-{label_id}"

        # 转换框坐标
        box = box.cpu().numpy()
        x_min, y_min, x_max, y_max = box
        
        w = x_max - x_min
        h = y_max - y_min
        
        detections.append({
            "bbox": [x_min, y_min, w, h],
            "label": label_name,
            "score": score
        })
        
    return detections

# --- D. 运行循环并展示 ---
print(f"\n{'='*20} 开始预测 {'='*20}")

for i, item in enumerate(dataset):
    image_data = item['image']
    image_id = item.get('image_id', f'demo_{i}')

    # 图片加载逻辑
    if isinstance(image_data, dict) and 'bytes' in image_data:
        try:
            image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
        except Exception as e:
            continue
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        continue
    
    print(f"\n📸 处理图片 [{i+1}/5] ID: {image_id}")
    
    detections = detect(image)
    
    if len(detections) == 0:
        print("   (未检测到高置信度物体)")
    else:
        for dt in detections:
            b = dt['bbox']
            print(f"   🎯 检测到: {dt['label']:<15} | 置信度: {dt['score']:.2f}")
            # 打印 x,y,w,h
            print(f"      BBox: <{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}>")
            
    # 画图
    if i < 5:
        draw = ImageDraw.Draw(image)
        # 尝试加载大一点的字体，如果没有就用默认的
        try:
            # Linux 系统常见字体路径，如果没有会报错回落到 except
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None # 使用默认极小字体

        for dt in detections:
            x, y, w, h = dt['bbox']
            # 画框
            draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
            
            # 画文字背景（防止文字看不清）
            text_content = f"{dt['label']} {dt['score']:.2f}"
            
            # 简单的文字背景计算
            if font:
                text_bbox = draw.textbbox((x, y), text_content, font=font)
            else:
                text_bbox = draw.textbbox((x, y), text_content) # 默认字体
            
            # 画一个红底白字的标签
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill='red')
            draw.text((x, y), text_content, fill='white', font=font)
        
        save_name = f"result_final_{i}.jpg" 
        image.save(save_name)
        print(f"   🖼️  结果已保存为 {save_name}")

print(f"\n{'='*20} 演示结束 {'='*20}")