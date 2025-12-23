import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.ops import box_iou

app = Flask(__name__)

# ================= 配置区域 =================
# 请确保这里是您的正确路径
MODEL_PATH = "/root/detr/detr-finetuned-coco-ver2"
# MODEL_PATH = "facebook/detr-resnet-50" # 备用

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

# ================= 模型加载 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model from: {MODEL_PATH} (Device: {device})...")

try:
    processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ 后端服务启动成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit() # 强制退出防止误导

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    threshold = float(request.form.get('threshold', 0.5))
    gt_str = request.form.get('gt_boxes', '').strip()

    try:
        image = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"error": "Invalid image file"}), 400

    # 1. 推理
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # 2. 后处理
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    # 3. 准备数据
    predictions = []
    pred_boxes_tensor = []
    
    # 先收集所有预测框，用于后续批量计算 IoU
    for box in results["boxes"]:
        pred_boxes_tensor.append(box.tolist())
    
    # 4. 处理真实框 (GT) 并计算 IoU
    has_gt = False
    per_box_ious = [None] * len(pred_boxes_tensor) # 默认全是 None
    mean_iou = "--"
    max_iou = "--"

    if gt_str and len(pred_boxes_tensor) > 0:
        try:
            gt_boxes = []
            for line in gt_str.split('\n'):
                # 兼容中文逗号
                line = line.replace('，', ',').strip()
                if not line: continue
                coords = [float(x) for x in line.split(',')]
                if len(coords) == 4:
                    gt_boxes.append(coords)
            
            if gt_boxes:
                has_gt = True
                gt_tensor = torch.tensor(gt_boxes).to(device)
                pred_tensor = torch.tensor(pred_boxes_tensor).to(device)
                
                # 计算矩阵: [预测框数量, 真实框数量]
                iou_matrix = box_iou(pred_tensor, gt_tensor)
                
                # 为每个预测框找到匹配度最高的真实框的 IoU 值
                # max_vals: 每个预测框对应的最大 IoU
                max_vals, _ = iou_matrix.max(dim=1)
                
                # 存入列表供前端表格使用
                per_box_ious = max_vals.tolist()
                
                # 计算全局指标
                mean_iou = f"{max_vals.mean().item():.4f}"
                max_iou = f"{max_vals.max().item():.4f}"
                
        except Exception as e:
            print(f"GT Parse Error: {e}")

    # 5. 组装最终返回数据
    for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box_cpu = box.tolist()
        label_idx = label.item()
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else str(label_idx)
        
        # 获取该框的 IoU (如果有)
        box_iou_val = per_box_ious[i] if has_gt else -1

        predictions.append({
            "label": label_name,
            "score": float(score),
            "box": box_cpu,
            "iou": box_iou_val # 新增字段：单框 IoU
        })

    return jsonify({
        "predictions": predictions,
        "iou": {
            "mean_iou": mean_iou,
            "max_iou": max_iou
        }
    })

if __name__ == '__main__':
    print("-" * 50)
    print("✅ 后端服务启动成功！")
    #print(f"👉 请在浏览器输入: http://127.0.0.1:5000")
    print("-" * 50)
    app.run(host='0.0.0.0', port=6006, debug=True)