import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.ops import box_iou, box_convert


MODEL_PATH = "/root/detr/detr-finetuned-coco-ver2"
IMAGE_PATH = "/root/detr/test_img/1c3eacc83e424d8b8eb4278116e56679.jpg"
OUTPUT_IMAGE_PATH = "result_visualization.jpg"
CONFIDENCE_THRESHOLD = 0.2

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

def plot_results(pil_img, prob, boxes, labels, gt_boxes=None, save_path="result.jpg"):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y1, 'GT', fontsize=10, bbox=dict(facecolor='lime', alpha=0.5))

    colors = ['r'] * len(boxes)
    for p, (xmin, ymin, xmax, ymax), l, c in zip(prob, boxes.tolist(), labels, colors):
        if p < CONFIDENCE_THRESHOLD:
            continue
            
        w, h = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
        
        l = COCO_80_CLASSES[l]
        text = f"{l}={p:.2f}"
        ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))


    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='red', label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {os.path.abspath(save_path)}")
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print(f"加载本地模型: {MODEL_PATH}")
    try:
        processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
        model.to(device)
    except OSError:
        print(f"❌ 错误: 找不到路径 {MODEL_PATH}")
        return

    image = Image.open(IMAGE_PATH)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
    )[0]

    pred_boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    print(f"\n正在绘制分析图...")
    plot_results(image, scores, pred_boxes, labels, None, save_path="/root/detr/test_result1")

if __name__ == "__main__":
    main()