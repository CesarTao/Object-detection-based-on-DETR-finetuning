import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.ops import box_iou, box_convert


MODEL_PATH = "/root/detr/detr-finetuned-coco-ver2"
IMAGE_PATH = "/root/autodl-tmp/coco2017/val2017/000000058655.jpg"
ANNOTATION_PATH = "/root/autodl-tmp/coco2017/annotations/instances_val2017.json"
OUTPUT_IMAGE_PATH = "result_visualization.jpg"
CONFIDENCE_THRESHOLD = 0.6


def load_ground_truth(json_path, image_path):
    print(f"正在读取标注文件...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    file_name = os.path.basename(image_path)
    try:
        target_id = int(file_name.split('.')[0])
    except ValueError:
        print("文件名格式错误")
        return torch.empty((0, 4))

    gt_boxes = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_id:
            gt_boxes.append(ann['bbox']) # [x, y, w, h]
    
    print(f"找到真实框数量: {len(gt_boxes)}")
    return torch.tensor(gt_boxes, dtype=torch.float32)

def plot_results(pil_img, prob, boxes, gt_boxes=None, save_path="result.jpg"):
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
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        if p < CONFIDENCE_THRESHOLD:
            continue
            
        w, h = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
        
        text = f'{p:.2f}'
        ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='lime', linestyle='--', label='Ground Truth'),
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

    gt_boxes_xywh = load_ground_truth(ANNOTATION_PATH, IMAGE_PATH)
    gt_boxes_xyxy = torch.empty((0, 4))
    
    if len(gt_boxes_xywh) > 0:
        gt_boxes_xyxy = box_convert(gt_boxes_xywh, in_fmt='xywh', out_fmt='xyxy')
    
    print(f"\n ===== 数据分析报告 =====")
    if len(pred_boxes) > 0 and len(gt_boxes_xyxy) > 0:
        iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)
        for i in range(len(pred_boxes)):
            best_iou, _ = iou_matrix[i].max(0)
            print(f"预测框 #{i+1} 置信度: {scores[i]:.2f} -> 最大 IoU: {best_iou:.4f}")
    else:
        print("无法计算 IoU (缺少预测框或真实框)")

    print(f"\n正在绘制分析图...")
    plot_results(image, scores, pred_boxes, gt_boxes_xyxy, save_path="/root/detr/test_result")

if __name__ == "__main__":
    main()