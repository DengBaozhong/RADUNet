# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:26:27 2025

@author: maxim
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP库（临时方案）
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
import random
import numpy as np
import torch
from RADUNet import LoveDADataset
from RADUNet import OpenEarthMapDataset
from RADUNet import ResNetUNet
from torch.utils.data import DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

random.seed(42)  # 固定Python随机种子
np.random.seed(42)  # 固定Numpy随机种子
torch.manual_seed(42)  # 固定PyTorch随机种子
torch.cuda.manual_seed_all(42)  # 固定所有CUDA随机种子
torch.backends.cudnn.deterministic = True  # 确保CuDNN结果可复现
torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动优化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total_pixel_acc = 0.0
    total_samples = 0
    
    # 初始化统计量
    conf_matrix = torch.zeros(num_classes, num_classes, device=device)
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs, _, _ = model(images)
            preds = outputs.argmax(dim=1)
            
            # 忽略未标注像素(类别0)
            valid_pixels = (masks != 0)
            
            # 1. 计算像素准确率(Pixel Accuracy)
            correct = (preds[valid_pixels] == masks[valid_pixels]).sum().item()
            total_pixels = valid_pixels.sum().item()
            total_pixel_acc += correct / total_pixels if total_pixels > 0 else 0
            
            # 2. 计算各类别的准确率(Class Accuracy)
            for cls in range(num_classes):
                cls_mask = (masks == cls)
                if cls == 0:  # 跳过未标注类别
                    continue
                cls_correct = (preds[cls_mask] == cls).sum().item()
                cls_total = cls_mask.sum().item()
                
                class_correct[cls] += cls_correct
                class_total[cls] += cls_total
            
            # 3. 更新混淆矩阵(用于计算mIoU)
            indices = masks[valid_pixels] * num_classes + preds[valid_pixels]
            counts = torch.bincount(indices, minlength=num_classes**2)
            conf_matrix += counts.reshape(num_classes, num_classes)
            
            total_samples += 1
    
    # 计算详细指标
    # 1. 像素准确率(全局)
    pixel_acc = total_pixel_acc / total_samples if total_samples > 0 else 0
    
    # 2. 类别准确率(Class Accuracy)和平均像素准确率(Mean Pixel Accuracy)
    class_acc = (class_correct / (class_total + 1e-6)).cpu().numpy()  # 避免除以零
    mean_pixel_acc = class_acc[1:].mean()  # 排除未标注类别(类别0)
    
    # 3. 计算各类别IoU和mIoU
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    miou = iou.mean().item()
    
    return {
        "pixel_accuracy": pixel_acc,
        "mean_pixel_accuracy": mean_pixel_acc,
        "class_accuracy": class_acc,
        "mean_iou": miou,
        "class_iou": iou.cpu().numpy(),
        "confusion_matrix": conf_matrix.cpu().numpy()
    }

    
def plot_metrics(metrics):
    class_names = ["Unlabeled", "Building", "Vegetation", "Water", "Barren", "Artificial"]
    
    # 创建子图
    plt.figure(figsize=(18, 6))
    
    # 1. 绘制各类别准确率
    plt.subplot(1, 3, 1)
    class_acc = metrics["class_accuracy"][1:]  # 排除未标注类别
    plt.bar(range(len(class_names)-1), class_acc)
    plt.xticks(range(len(class_names)-1), class_names[1:], rotation=45)
    plt.title("Class Accuracy (Excluding Unlabeled)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    # 2. 绘制各类别IoU
    plt.subplot(1, 3, 2)
    class_iou = metrics["class_iou"][1:]  # 排除未标注类别
    plt.bar(range(len(class_names)-1), class_iou)
    plt.xticks(range(len(class_names)-1), class_names[1:], rotation=45)
    plt.title("IoU per Class (Excluding Unlabeled)")
    plt.ylabel("IoU Score")
    plt.ylim(0, 1)
    
    # 3. 打印总体指标
    plt.subplot(1, 3, 3)
    plt.axis('off')
    text = (
        f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n"
        f"Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.4f}\n"
        f"Mean IoU: {metrics['mean_iou']:.4f}\n"
    )
    plt.text(0.1, 0.5, text, fontsize=12)
    plt.title("Overall Metrics")
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".1f", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def save_metrics_to_file(metrics, fold, filename="evaluation_results.txt"):
    """将评估结果保存到文件"""
    class_names = ["Unlabeled", "Building", "Vegetation", "Water", "Barren", "Artificial"]
    
    with open(filename, 'a') as f:
        f.write(f"\n=== Fold {fold} Evaluation Results ===\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        
        # 写入各类别准确率
        f.write("\nClass Accuracy (Excluding Unlabeled):\n")
        for i in range(1, len(class_names)):
            f.write(f"{class_names[i]}: {metrics['class_accuracy'][i]:.4f}\n")
        
        # 写入各类别IoU
        f.write("\nClass IoU (Excluding Unlabeled):\n")
        for i in range(1, len(class_names)):
            f.write(f"{class_names[i]}: {metrics['class_iou'][i]:.4f}\n")
        
        # 写入带标签的混淆矩阵
        f.write("\nConfusion Matrix (行=真实标签, 列=预测标签):\n")
        
        # 首先写入列标签（预测类别）
        header = "\t" + "\t".join(class_names) + "\n"
        f.write(header)
        
        # 写入每行数据（真实类别标签+混淆矩阵数据）
        conf_matrix = metrics["confusion_matrix"]
        for i in range(len(class_names)):
            row_label = class_names[i]
            row_data = "\t".join(map(str, conf_matrix[i]))
            f.write(f"{row_label}\t{row_data}\n")
        
        f.write("\n")

if __name__ == '__main__':
    print("脚本开始执行...")
    torch.cuda.empty_cache()  # 清理残留的显存占用
    print(f"当前可用显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    print(f"BF16支持: {torch.cuda.is_bf16_supported()}")
    print(f"TF32支持: {torch.backends.cuda.matmul.allow_tf32}")  # RTX 40系列默认启用
    # 数据增强
    IMG_SIZE = 1024 // 2
    
    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    # 加载数据集
    print("\n正在加载数据集...")
    # 创建独立的数据集对象（关键修改点）
    # 示例：兼容随机分割的写法（根据需求选择）
    img_dir_loveda = './LoveDA/images_png'
    mask_dir_loveda = './LoveDA/masks_png'
    img_dir_openearthmap = './OpenEarthMap/images'
    mask_dir_openearthmap = './OpenEarthMap/masks'
    print(f"检查图像目录是否存在: {os.path.exists(img_dir_loveda)}")
    print(f"检查掩码目录是否存在: {os.path.exists(mask_dir_loveda)}")
    print(f"检查图像目录是否存在: {os.path.exists(img_dir_openearthmap)}")
    print(f"检查掩码目录是否存在: {os.path.exists(mask_dir_openearthmap)}")
    dataset_loveda = LoveDADataset(img_dir_loveda, mask_dir_loveda)
    dataset_openearthmap = OpenEarthMapDataset(img_dir_openearthmap, mask_dir_openearthmap)
    print(f"LoveDA 数据集样本数: {len(dataset_loveda)}")
    print(f"OpenEarthMap 数据集样本数: {len(dataset_openearthmap)}")

    # 合并数据集
    full_dataset = ConcatDataset([dataset_loveda, dataset_openearthmap])
    
    # 初始化K-Fold交叉验证
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # 清空或创建结果文件
    with open("evaluation_results.txt", 'w') as f:
        f.write("Cross-Validation Evaluation Results\n")
        f.write("===================================\n")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n=== 开始第 {fold+1}/{k_folds} 折交叉验证 ===")
        
        # 创建子集并设置transform
        train_subsets = []
        val_subsets = []
        
        # 处理ConcatDataset中的每个子数据集
        for dataset in full_dataset.datasets:
            # 获取该子数据集在原始full_dataset中的索引范围
            start_idx = 0 if dataset == full_dataset.datasets[0] else len(full_dataset.datasets[0])
            end_idx = start_idx + len(dataset)
            sub_val_ids = [i - start_idx for i in val_ids if start_idx <= i < end_idx]
            val_subset = torch.utils.data.Subset(dataset, sub_val_ids)
            val_subset.dataset.transform = val_transform
            val_subsets.append(val_subset)
        
        val_dataset = torch.utils.data.ConcatDataset(val_subsets)
        
        # 创建数据加载器
        batch_size = 24
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)
        
        # 检查是否存在之前训练过的模型
        model_path = f'best_model_fold{fold+1}.pth'
        n_classes = 6   # 我们的6个类别
        
        print(f"正在加载模型 {model_path}...")
        model = ResNetUNet(n_classes=n_classes, pretrained=False).to(device)  # 不加载预训练权重
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("模型加载成功")
            
        # 评估模型
        print(f"\n正在评估模型 {model_path}...")
        metrics = evaluate_model(model, val_loader, device, n_classes)
        plot_metrics(metrics)
        
        # 打印评估结果
        print(f"\nFold {fold+1} Evaluation Results:")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")

        # 可视化混淆矩阵
        class_names = ["Unlabeled", "Building", "Vegetation", "Water", "Barren", "Artificial"]
        plot_confusion_matrix(metrics["confusion_matrix"], class_names)
        
        # 保存评估结果到文件
        save_metrics_to_file(metrics, fold+1)
        print(f"Fold {fold+1} 评估结果已保存到 evaluation_results.txt")
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
    
    