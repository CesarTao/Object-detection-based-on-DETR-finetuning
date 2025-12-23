import os
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os
import torch
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from io import BytesIO
import albumentations as A
from torch.optim import AdamW

# ================= 1. 配置区域 =================
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

LOCAL_DATA_DIR = "/root/autodl-tmp/coco_full"
OUTPUT_DIR = "./detr-finetuned-coco"
#CHECKPOINT = "facebook/detr-resnet-50"
CHECKPOINT = "/root/detr/detr-resnet-50-coco-80class"


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

# ================= 2. 数据加载 =================
print("加载数据...")
data_files = {
    "train": os.path.join(LOCAL_DATA_DIR, "data/train-*.parquet"), 
    "validation": os.path.join(LOCAL_DATA_DIR, "data/val-*.parquet")
}
dataset = load_dataset("parquet", data_files=data_files)

print(f"训练集: {len(dataset['train'])} | 验证集: {len(dataset['validation'])}")

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

train_augment = A.Compose([
    A.HorizontalFlip(p=0.5), # 50% 概率水平翻转 (框也会自动翻)
    A.RandomResizedCrop(size=(800, 800), scale=(0.7, 1.0), p=0.5), # 随机裁剪/缩放
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),# 像素级变换,颜色、亮度
    A.GaussNoise(p=0.1),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.1))

from transformers import TrainerCallback
import torch

class EMACallback(TrainerCallback):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters() 
            if param.requires_grad
        }

    def on_step_end(self, args, state, control, model, **kwargs):
        decay = self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    if self.shadow_params[name].device != param.device:
                        self.shadow_params[name] = self.shadow_params[name].to(param.device)
                    
                    self.shadow_params[name].lerp_(param.data, 1.0 - decay)

    def on_train_end(self, args, state, control, model, **kwargs):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name].to(param.device))
        print("EMA 权重应用完毕")


def train_transforms(batch):
    pixel_values = []
    labels = []
    
    for i in range(len(batch["image"])):
        img_data = batch["image"][i]
        image = Image.open(BytesIO(img_data['bytes'])).convert("RGB") if isinstance(img_data, dict) else img_data.convert("RGB")
        image_np = np.array(image)
        
        objects = batch["objects"][i]
        original_boxes = []
        original_cats = []
        
        if len(objects['bbox']) > 0:
            for box, cat_id in zip(objects['bbox'], objects['category']):
                x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                w = x_max - x_min
                h = y_max - y_min
                
                if w <= 1 or h <= 1: continue
                
                cid = int(cat_id)
                if cid >= 80: cid = cid % 80
                
                original_boxes.append([x_min, y_min, w, h])
                original_cats.append(cid)
        
        if len(original_boxes) > 0:
            transformed = train_augment(
                image=image_np, 
                bboxes=original_boxes, 
                category_ids=original_cats
            )
            aug_image_np = transformed['image']
            aug_boxes = transformed['bboxes']
            aug_cats = transformed['category_ids']
        else:
            aug_image_np = image_np
            aug_boxes = []
            aug_cats = []

        aug_image_pil = Image.fromarray(aug_image_np)
        
        target_anns = []
        for box, cat_id in zip(aug_boxes, aug_cats):
            x, y, w, h = box
            target_anns.append({
                "image_id": batch["image_id"][i],
                "category_id": cat_id,
                "isCrowd": 0,
                "area": w * h,
                "bbox": [x, y, w, h]
            })
            
        encoding = image_processor(
            images=aug_image_pil, 
            annotations={'image_id': batch["image_id"][i], 'annotations': target_anns}, 
            return_tensors="pt"
        )
        
        pixel_values.append(encoding["pixel_values"].squeeze())
        labels.append(encoding["labels"][0])
        
    return {"pixel_values": pixel_values, "labels": labels}

def val_transforms(batch):
    pixel_values = []
    labels = []
    
    for i in range(len(batch["image"])):
        img_data = batch["image"][i]
        image = Image.open(BytesIO(img_data['bytes'])).convert("RGB") if isinstance(img_data, dict) else img_data.convert("RGB")
        
        target_anns = []
        objects = batch["objects"][i]
        
        if len(objects['bbox']) > 0:
            for box, cat_id in zip(objects['bbox'], objects['category']):
                x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                w = x_max - x_min
                h = y_max - y_min
                
                if w <= 1 or h <= 1: continue
                
                cid = int(cat_id)
                if cid >= 80: cid = cid % 80
                
                target_anns.append({
                    "image_id": batch["image_id"][i],
                    "category_id": cid,
                    "isCrowd": 0,
                    "area": w * h,
                    "bbox": [x_min, y_min, w, h]
                })
        
        formatted_annotations = {'image_id': batch["image_id"][i], 'annotations': target_anns}
        encoding = image_processor(
            images=image, 
            annotations=formatted_annotations, 
            return_tensors="pt"
        )
        
        pixel_values.append(encoding["pixel_values"].squeeze())
        labels.append(encoding["labels"][0])
        
    return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"], 
        "pixel_mask": encoding["pixel_mask"], 
        "labels": labels
    }


train_dataset = dataset["train"].with_transform(train_transforms)
eval_dataset = dataset["validation"].with_transform(val_transforms)

# ================= 3. 加载模型 =================
model = DetrForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=False
)

# ================= 4. 训练参数 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # 显存够可以开 8
    gradient_accumulation_steps=2, 
    
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2, 
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4, 
    remove_unused_columns=False,
)
param_dicts = [
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": 1e-5,
    },
    {
        # 找出所有名字里 不包含 "backbone" 的参数 (即 Transformer 和分类头)
        "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
        "lr": 1e-4,
    },
]

optimizer = AdamW(param_dicts, weight_decay=1e-4)

# ================= 5. 开始训练 =================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # 加上验证集
    tokenizer=image_processor,
    optimizers=(optimizer, None),
    callbacks=[EMACallback(model, decay=0.999)]
)

print("开始微调...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
image_processor.save_pretrained(OUTPUT_DIR)
print(f"训练完成！模型已保存至 {OUTPUT_DIR}")