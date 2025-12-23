import os
import torch
import json
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.ops import box_iou, box_convert
from tqdm import tqdm


MODEL_PATH = "/root/detr/detr-finetuned-coco-ver2"
VAL_DIR = "/root/autodl-tmp/coco2017/val2017"
ANNOTATION_PATH = "/root/autodl-tmp/coco2017/annotations/instances_val2017.json"
CONFIDENCE_THRESHOLD = 0.95 


def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_map:
            gt_map[img_id] = []
        # COCO 格式: [x, y, w, h]
        gt_map[img_id].append(ann['bbox'])
    
    print(f"标注加载完毕，共包含 {len(gt_map)} 张图片的标注信息。")
    return gt_map

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    try:
        processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except OSError:
        print(f"错误: 找不到模型路径 {MODEL_PATH}")
        return

    gt_map = load_coco_annotations(ANNOTATION_PATH)

    image_files = [f for f in os.listdir(VAL_DIR) if f.endswith('.jpg')]
    print(f"验证集共有 {len(image_files)} 张图片，开始评估...")

    total_images_processed = 0
    total_iou_sum = 0.0
    total_matched_boxes = 0

    for img_file in tqdm(image_files, desc="Running Inference"):
        img_path = os.path.join(VAL_DIR, img_file)
        
        try:
            img_id = int(img_file.split('.')[0])
        except ValueError:
            continue

        if img_id not in gt_map:
            continue
            
        gt_boxes_xywh = torch.tensor(gt_map[img_id], dtype=torch.float32)
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            print(f"读取图片 {img_file} 失败: {e}")
            continue

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
        )[0]

        pred_boxes = results["boxes"].cpu()
        
        if len(pred_boxes) == 0:
            continue

        gt_boxes_xyxy = box_convert(gt_boxes_xywh, in_fmt='xywh', out_fmt='xyxy')
        
        iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)
        
        max_ious, _ = iou_matrix.max(dim=1) 

        current_img_avg_iou = max_ious.mean().item()
        print(f"[图片: {img_file}] 检测框数: {len(pred_boxes)} | 本图平均 IoU: {current_img_avg_iou:.4f}")
        
        total_iou_sum += max_ious.sum().item()
        total_matched_boxes += len(max_ious)
        total_images_processed += 1

    print("\n" + "="*40)
    print("最终结果")
    print("="*40)
    print(f"已处理图片数量: {total_images_processed}")
    print(f"检测到的物体总数: {total_matched_boxes}")
    
    if total_matched_boxes > 0:
        avg_iou = total_iou_sum / total_matched_boxes
        print(f"平均 IoU : {avg_iou:.4f}")
    else:
        print("未检测到任何有效物体。")
    print("="*40)

if __name__ == "__main__":
    main()