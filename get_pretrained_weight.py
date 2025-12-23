import os
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import os
from transformers import DetrForObjectDetection, DetrConfig
from transformers import DetrImageProcessor

# ================= 配置 =================
CHECKPOINT = "facebook/detr-resnet-50"
SAVE_PATH = "./detr-resnet-50-coco-80class"

COCO_CLASSES_80 = [
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

def perform_surgery():
    old_model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
    old_weights = old_model.class_labels_classifier.weight.data # [92, 256]
    old_bias = old_model.class_labels_classifier.bias.data      # [92]
    
    old_id2label = old_model.config.id2label
    old_label2id = {v: k for k, v in old_id2label.items()}
    
    new_id2label = {i: label for i, label in enumerate(COCO_CLASSES_80)}
    new_label2id = {label: i for i, label in new_id2label.items()}
    
    new_model = DetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        num_labels=80,
        id2label=new_id2label,
        label2id=new_label2id,
        ignore_mismatched_sizes=True 
    )
    
    new_weights = new_model.class_labels_classifier.weight.data # [81, 256]
    new_bias = new_model.class_labels_classifier.bias.data      # [81]
    
    transferred_count = 0
    
    for new_idx, class_name in enumerate(COCO_CLASSES_80):
        if class_name in old_label2id:
            old_idx = old_label2id[class_name]
            
            new_weights[new_idx] = old_weights[old_idx]
            new_bias[new_idx] = old_bias[old_idx]
            transferred_count += 1
        else:
            print(f"警告: 在原模型中没找到 '{class_name}'")


    new_weights[-1] = old_weights[-1] 
    new_bias[-1] = old_bias[-1]
    
    print(f"💾 正在保存到: {SAVE_PATH} ...")
    processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    processor.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    perform_surgery()