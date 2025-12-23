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
from torchvision.ops import batched_nms

# ================= 配置区域 =================
LOCAL_DATA_DIR = "/root/autodl-tmp/coco_full"
MODEL_PATH = "/root/detr/detr-finetuned-coco" 
#MODEL_PATH = "/root/detr/detr-resnet-50-coco-80class"
# ===========================================

print("🚀 正在初始化环境...")

try:
    print(f"正在从 {LOCAL_DATA_DIR} 加载数据...")
    data_files = {"validation": os.path.join(LOCAL_DATA_DIR, "data/val-*.parquet")}
    dataset = load_dataset("parquet", data_files=data_files, split="validation[:5]")
    print(f"数据加载成功！共 {len(dataset)} 张图片。")
except Exception as e:
    print(f"数据加载失败: {e}")
    dataset = load_dataset("detection-datasets/coco", split="validation[:5]", streaming=True)


print(f"正在加载模型: {MODEL_PATH}")
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"模型就绪！运行设备: {device}")


COCO_80_CLASSES = [
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



def detect(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    if boxes.shape[0] == 0:
        return []

    # 使用 batched_nms
    # 作用：只有当“类别相同”且“重叠度高”时才抑制。
    keep_indices = batched_nms(boxes, scores, labels, iou_threshold=0.3)
    
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]

    detections = []
    for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
        score = score.item()
        label_id = label.item()
        
        if 0 <= label_id < len(COCO_80_CLASSES):
            label_name = COCO_80_CLASSES[label_id]
        else:
            label_name = f"Unknown-{label_id}"

        box = box.cpu().numpy()
        x_min, y_min, x_max, y_max = box
        
        detections.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "label": label_name,
            "score": score,
            "label_id": label_id
        })
        
    return detections

print(f"\n{'='*20} 开始预测 {'='*20}")

for i, item in enumerate(dataset):
    image_data = item['image']
    image_id = item.get('image_id', f'demo_{i}')

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
            print(f"检测到: {dt['label']:<15} | 置信度: {dt['score']:.2f}")
            print(f"   ID: {dt['label_id']:<15} | BBox 原始数据: {b}")
            
    if i < 5:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None 

        for dt in detections:
            x_min, y_min, x_max, y_max = dt['bbox']
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            
            text_content = f"{dt['label']} {dt['score']:.2f}"
            if font:
                text_bbox = draw.textbbox((x_min, y_min), text_content, font=font)
            else:
                text_bbox = draw.textbbox((x_min, y_min), text_content)
            
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill='red')
            draw.text((x_min, y_min), text_content, fill='white', font=font)
        
        save_name = f"result_preview_{i}.jpg" 
        image.save(save_name)
        print(f"结果已保存为 {save_name}")

print(f"\n{'='*20} 演示结束 {'='*20}")