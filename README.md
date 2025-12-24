**基于DETR微调的目标检测任务**  
源项目：https://github.com/facebookresearch/detr?tab=readme-ov-file  
数据集：coco2017

根目录下  
app.py: 网页端的可视化预测工具  
coco_test.py: 检测coco数据集  
eval.py: 针对数据集的评估  
finetune.py: 在detr模型与预权重基础上的微调  
get_pretrained_weight：对官方预权重进行处理，使其能够适用于80类标签的detr模型  
infer.py:针对单张图片的可视化预测  
inference&inference2.py: 针对数据集的预测，前五张图片可视化  
visualize.py: 针对单张图片的预测、评估和可视化  
test：一些没用的测试脚本  
其他文件均来自源项目  

extra requirements:  
torch
torchvision
transformers
timm
numpy
scipy
Pillow
matplotlib
Flask
tqdm
