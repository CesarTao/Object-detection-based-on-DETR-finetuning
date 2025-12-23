import os
# 1. 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. (可选) 强制指定缓存到数据盘，防止系统盘爆满
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"


import os
import torch
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

# ================= 1. 基础配置 =================
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
LOCAL_DATA_DIR = "/root/autodl-tmp/coco_full"
CHECKPOINT = "facebook/detr-resnet-50"
OUTPUT_DIR = "./detr-final-test"

# COCO 80 类 (0-79)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
id2label = {i: label for i, label in enumerate(COCO_CLASSES)}
label2id = {label: i for i, label in id2label.items()}

# ================= 2. 数据准备 =================
print("🚀 加载数据...")
data_files = {"train": os.path.join(LOCAL_DATA_DIR, "data/train-*.parquet")}
# ⚠️ 只取 10 张图过拟合
full_dataset = load_dataset("parquet", data_files=data_files, split="train[:10]")
print(f"🧪 Final Test: 使用 {len(full_dataset)} 张图片")

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

def train_transforms(batch):
    pixel_values = []
    labels = []
    
    for i in range(len(batch["image"])):
        img_data = batch["image"][i]
        image = Image.open(BytesIO(img_data['bytes'])).convert("RGB") if isinstance(img_data, dict) else img_data.convert("RGB")
        
        target_anns = []
        objects = batch["objects"][i]
        
        if len(objects['bbox']) > 0:
            for box, cat_id in zip(objects['bbox'], objects['category']):
                # 1. 坐标处理: [xmin, ymin, xmax, ymax]
                x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                w = x_max - x_min
                h = y_max - y_min
                
                if w <= 1 or h <= 1: continue
                
                # 2. ID 处理: 直接透传 (因为数据已经是 0-79 了)
                # 你的 "ID 0" 对应 "person"，不需要映射
                cid = int(cat_id)
                if cid >= 80: 
                    # 防御性编程：万一有脏数据
                    cid = cid % 80
                
                target_anns.append({
                    "image_id": batch["image_id"][i],
                    "category_id": cid, 
                    "isCrowd": 0,
                    "area": w * h,
                    "bbox": [x_min, y_min, w, h] # xywh
                })
        
        # 3. 字典包装
        formatted_annotations = {'image_id': batch["image_id"][i], 'annotations': target_anns}
        encoding = image_processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        
        pixel_values.append(encoding["pixel_values"].squeeze())
        labels.append(encoding["labels"][0])
        
    return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}

train_dataset = full_dataset.with_transform(train_transforms)

# ================= 3. 训练配置 =================
model = DetrForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    # 🔥 策略：大火猛炒
    num_train_epochs=300,          
    learning_rate=1e-4,           
    weight_decay=0.0,             
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    tokenizer=image_processor,
)

# ================= 4. 运行 =================
print("\n🔥 开始最终训练...")
trainer.train()

# ================= 5. 验证 (显微镜模式) =================
print(f"\n{'='*20} 最终验证 {'='*20}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 取第一张图
sample = full_dataset[0]
image = Image.open(BytesIO(sample['image']['bytes'])).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]]).to(device)

# 🚨 阈值设为 0.0，查看所有可能的预测
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

# 按照分数排序，只看前 10 个
if len(scores) > 0:
    topk = min(10, len(scores))
    indices = torch.topk(scores, topk).indices
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]

draw = ImageDraw.Draw(image)
print("📸 预测 Top 10 (无阈值):")

found_valid = False
for score, label, box in zip(scores, labels, boxes):
    label_str = COCO_CLASSES[label.item()]
    box = box.cpu().numpy()
    
    # 打印 log
    print(f"  👉 {label_str} | Score: {score:.4f} | Box: {box.astype(int)}")
    
    # 只要分数 > 0.1 就画框
    if score > 0.1:
        found_valid = True
        draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1]), f"{label_str} {score:.2f}", fill='red')

if not found_valid:
    print("\n⚠️ 前 10 个结果分数都很低 (<0.1)。模型可能还在纠结，或者需要更久训练。")
else:
    print("\n✅ 看到高分结果了！")

image.save("final_success.jpg")
print("🖼️ 结果已保存为 final_success.jpg")