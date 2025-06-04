# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 07:11:17 2025

@author: maxim
"""

import os
import numpy as np
import cv2
import csv
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_class_area(prediction):
    unique, counts = np.unique(prediction, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if 0 in class_counts:
        total_pixels = prediction.size - class_counts[0]
    else:
        total_pixels = prediction.size
    
    class_ratios = {}
    for class_id in range(1, 6):
        class_pixels = class_counts.get(class_id, 0)
        class_ratios[class_id] = class_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return class_ratios

def generate_visualization(prediction, output_path):
    """生成可视化图片"""
    colormap = {
        1: [170, 170, 170],  # 建筑
        2: [30, 130, 20],    # 植被
        3: [60, 190, 200],    # 水域
        4: [150, 160, 110],   # 裸地
        5: [60, 60, 60],      # 人工表面
        0: [0, 0, 0]          # 未标注
    }
    class_names = ['Unlabelled', 'Building', 'Vegetation', 'Water', 'Barren', 'Artificial']
    
    colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored[prediction == label] = color
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(colored)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    for label in range(6):
        plt.plot([], [], 's', color=np.array(colormap[label])/255, label=class_names[label])
    plt.legend(loc='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)  # 降低DPI提高速度
    plt.close()

def process_masks(pred_dir, output_csv, visualize=False):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'Building', 'Vegetation', 'Water', 'Barren', 'Artificial'])
        
        pred_files = [f for f in os.listdir(pred_dir) if f.startswith('pred_') and f.lower().endswith('.png')]
        
        # 如果需要可视化，创建目录
        if visualize:
            vis_dir = os.path.join(pred_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        for pred_file in tqdm(pred_files, desc="Processing"):
            try:
                original_name = pred_file[5:]
                pred_path = os.path.join(pred_dir, pred_file)
                prediction = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                ratios = calculate_class_area(prediction)
                writer.writerow([
                    original_name,
                    ratios.get(1, 0),
                    ratios.get(2, 0),
                    ratios.get(3, 0),
                    ratios.get(4, 0),
                    ratios.get(5, 0)
                ])
                
                if visualize:
                    vis_path = os.path.join(vis_dir, f'vis_{original_name}')
                    generate_visualization(prediction, vis_path)
                    
            except Exception as e:
                print(f"\nError in {pred_file}: {str(e)}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    visualize = False
    
    pred_directory = './Province/pred'
    output_csv = './Province/classification_results.csv'
    
    print(f"开始处理，可视化模式: {'开启' if visualize else '关闭'}")
    process_masks(pred_directory, output_csv, visualize=visualize)
    print(f"\n处理完成！结果已保存到: {output_csv}")
