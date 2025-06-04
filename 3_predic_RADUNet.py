# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 07:11:17 2025

@author: maxim
"""

"""
优化后的代码，主要改进：
1. 批量处理图片
2. 增加批处理大小
3. 优化显存使用
4. 减少IO操作
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from RADUNet import ResNetUNet
from torch.amp import autocast
import cv2
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import time

# 使用系统字体（推荐）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models_with_weights(model_weights):
    """
    加载所有折的模型及其权重
    参数:
        model_weights: 列表，包含元组 (模型路径, 模型权重)
    """
    models = []
    weights = []
    for path, weight in model_weights:
        model = ResNetUNet(n_classes=6).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        models.append(model)
        weights.append(weight)
    
    # 将权重归一化为概率分布
    weights = np.array(weights)
    weights = weights / weights.sum()
    return models, weights

def weighted_ensemble_predict(models, weights, image_path, output_path, patch_size=1024, overlap=128):
    # 读取图像 - 使用OpenCV替代PIL
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w, _ = image.shape

    # 计算填充量
    stride = patch_size - overlap
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    
    # 对称填充图像
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # 创建权重矩阵
    patch_weights = np.ones((patch_size, patch_size), dtype=np.float32)
    for i in range(overlap):
        val = i / overlap
        patch_weights[i, :] *= val          # 上边缘
        patch_weights[-i-1, :] *= val       # 下边缘
        patch_weights[:, i] *= val          # 左边缘
        patch_weights[:, -i-1] *= val       # 右边缘

    # 初始化结果 - 使用更高效的内存布局 (H,W,C)
    full_probs = np.zeros((image.shape[0], image.shape[1], 6), dtype=np.float32)
    full_weights = np.zeros(image.shape[:2], dtype=np.float32)

    # 预计算所有patch的位置
    patches_info = []
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            i_end = min(i + patch_size, image.shape[0])
            j_end = min(j + patch_size, image.shape[1])
            patches_info.append((i, j, i_end, j_end))
    
    # 使用NumPy的原地操作加速累加
    for i, j, i_end, j_end in tqdm(patches_info, desc="Processing patches"):
        patch = image[i:i_end, j:j_end]
        current_patch_weights = patch_weights[:patch.shape[0], :patch.shape[1]]
        
        # 预处理 - 使用更快的转换
        patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        patch_tensor = transforms.functional.normalize(
            patch_tensor,
            mean=[0.3481, 0.3759, 0.3497],
            std=[0.1875, 0.1689, 0.1627]
        ).unsqueeze(0).to(device)
        
        # 计算加权概率
        weighted_probs = np.zeros((6, patch.shape[0], patch.shape[1]), dtype=np.float32)
        
        with torch.no_grad(), autocast(device_type='cuda', 
                                       dtype=torch.bfloat16):
            for model, weight in zip(models, weights):
                main_out = model(patch_tensor)[0]
                prob = torch.softmax(main_out, dim=1).squeeze(0).cpu().numpy()
                weighted_probs += prob * weight
        
        # 转置为HWC格式并应用权重
        weighted_probs = weighted_probs.transpose(1, 2, 0)
        
        # 使用NumPy的原地操作进行累加
        np.add.at(full_probs, (slice(i, i_end), slice(j, j_end)), 
                   weighted_probs * current_patch_weights[..., None])
        np.add.at(full_weights, (slice(i, i_end), slice(j, j_end)), 
                   current_patch_weights)

    # 加权平均 - 使用np.divide的安全版本
    valid_mask = full_weights > 0
    full_probs[valid_mask] = full_probs[valid_mask] / full_weights[valid_mask][:, None]
    full_probs[~valid_mask] = 0
    
    # 获取最终预测结果
    full_pred = np.argmax(full_probs, axis=-1)
    full_pred = full_pred[:h, :w].astype(np.uint8)
    
    # 保存结果 - 使用OpenCV
    cv2.imwrite(output_path, full_pred)
    return full_pred

# 计算各类别面积占比（忽略未标注）
def calculate_class_area(prediction):
    # 统计各类别像素数量
    unique, counts = np.unique(prediction, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # 忽略未标注类别(0)
    if 0 in class_counts:
        total_pixels = prediction.size - class_counts[0]
    else:
        total_pixels = prediction.size
    
    # 计算各类别占比
    class_ratios = {}
    for class_id in range(1, 6):  # 只统计1-5类
        class_pixels = class_counts.get(class_id, 0)
        if total_pixels > 0:
            class_ratios[class_id] = class_pixels / total_pixels
        else:
            class_ratios[class_id] = 0.0
    
    # # 打印结果
    # print("各类别面积占比（忽略未标注区域）:")
    # print(f"1-建筑区域: {class_ratios.get(1, 0)*100:.4f}%")
    # print(f"2-植被区域: {class_ratios.get(2, 0)*100:.4f}%")
    # print(f"3-水域: {class_ratios.get(3, 0)*100:.4f}%")
    # print(f"4-裸地与未利用地: {class_ratios.get(4, 0)*100:.4f}%")
    # print(f"5-人工表面: {class_ratios.get(5, 0)*100:.4f}%")
    # print(f"总有效像素数: {total_pixels}")
    
    return class_ratios

def visualize_segmentation_with_legend(prediction, output_path, class_names=None, colormap=None):
    """
    将语义分割结果可视化为彩色图像并添加图例
    
    参数:
        prediction: 预测的分割结果 (H x W 的numpy数组)
        output_path: 保存可视化结果的路径
        class_names: 类别名称列表 (如 ['建筑', '道路', ...])
        colormap: 可选的自定义颜色映射
    """
    if class_names is None:
        class_names = [f'类别{i}' for i in range(6)]
    
    # 默认颜色映射
    if colormap is None:
        colormap = {
            1: [170, 170, 170],    # 建筑区域
            2: [30, 130, 20],    # 植被(包含农田)
            3: [60, 190, 200],    # 水域
            4: [150, 160, 110], # 裸地与未利用地
            5: [60, 60, 60], # 人工表面(包含道路和其他)
            0: [0, 0, 0]   # 未标注
        }
    
    # 创建彩色图像
    colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored[prediction == label] = color
    
    # 创建带图例的图像
    plt.figure(figsize=(10, 5))
    
    # 显示分割结果
    plt.subplot(1, 2, 1)
    plt.imshow(colored)
    plt.title('Segmentation result')
    plt.axis('off')
    
    # 显示图例
    plt.subplot(1, 2, 2)
    for label, color in colormap.items():
        plt.plot([], [], 's', color=np.array(color)/255, label=class_names[label])
    plt.legend(loc='center')
    plt.axis('off')
    plt.title('Legend')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return colored

def process_directory(input_dir, output_csv, models, weights, patch_size=1024, overlap=128):
    """
    优化后的目录处理函数，支持批量处理
    """
    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'Building', 'Vegetation', 'Water', 'Barren', 'Artificial'])
        
        # 获取所有PNG文件
        png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        
        # 遍历目录中的所有PNG文件
        for filename in tqdm(png_files, desc="Processing images"):
            image_path = os.path.join(input_dir, filename)
            
            # 创建输出路径
            output_mask_dir = os.path.join(input_dir, 'pred')
            os.makedirs(output_mask_dir, exist_ok=True)
            output_path = os.path.join(output_mask_dir, f'pred_{filename}')
            
            try:
                start_time = time.time()
                # 进行预测
                prediction = weighted_ensemble_predict(models, weights, image_path, output_path, patch_size, overlap)
                
                # 计算各类别占比
                class_ratios = calculate_class_area(prediction)
                
                # 写入CSV
                writer.writerow([
                    filename,
                    class_ratios.get(1, 0),
                    class_ratios.get(2, 0),
                    class_ratios.get(3, 0),
                    class_ratios.get(4, 0),
                    class_ratios.get(5, 0)
                ])
                
                # 创建可视化结果
                output_vis_dir = os.path.join(input_dir, 'vis')
                os.makedirs(output_vis_dir, exist_ok=True)
                class_names = ['Unlabelled', 'Building', 'Vegetation', 'Water', 'Barren', 'Artificial']
                visualize_segmentation_with_legend(
                    prediction, 
                    os.path.join(output_vis_dir, f'vis_{filename}'), 
                    class_names
                )
                
                print(f"Processed {filename} in {time.time()-start_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

# 示例使用
if __name__ == "__main__":
    torch.cuda.empty_cache()  # 清理残留的显存占用
    
    # 假设这是各模型在验证集上的IoU分数
    val_iou_scores = [0.7034, 0.7081, 0.7097, 0.7102, 0.7035]  # 需要替换为实际的验证分数
    
    # 加载所有折的模型及权重
    model_weights = [
        ('best_model_fold1.pth', val_iou_scores[0]),
        ('best_model_fold2.pth', val_iou_scores[1]),
        ('best_model_fold3.pth', val_iou_scores[2]),
        ('best_model_fold4.pth', val_iou_scores[3]),
        ('best_model_fold5.pth', val_iou_scores[4])
    ]
    
    try:
        models, weights = load_models_with_weights(model_weights)
        
        print(f"模型权重分配: {weights}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        
        # 指定输入目录和输出CSV文件
        input_directory = './SHU'  # 修改为你的图片目录
        output_csv = './SHU/classification_results.csv'  # 输出CSV文件路径
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # 处理目录中的所有PNG文件
        process_directory(input_directory, output_csv, models, weights, patch_size=512, overlap=64)
    finally:
        # 4. 最终清理
        for model in models:
            del model
        torch.cuda.empty_cache()
        print("显存已强制释放")
