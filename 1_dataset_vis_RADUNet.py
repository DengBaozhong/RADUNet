# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:26:27 2025

@author: maxim
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP库（临时方案）
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
import numpy as np
import torch
from RADUNet import LoveDADataset
from RADUNet import OpenEarthMapDataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
import multiprocessing as mp
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 使用系统字体（推荐）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# s手动忽略未标注类别（0）
valid_classes = [1, 2, 3, 4, 5]
class_names = {
    1: "建筑区域",
    2: "植被",
    3: "水域",
    4: "裸地",
    5: "人工表面"
}

class StatsBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size=64, num_workers=8):
        super().__init__(RandomSampler(dataset), batch_size, drop_last=False)
        self.dataset = dataset
        self.num_workers = num_workers
    
    def compute_batch_stats(self, batch_indices):
        batch_rgb_sum = np.zeros(3)
        batch_rgb_sum_sq = np.zeros(3)
        batch_pixels = 0
        class_stats = defaultdict(lambda: {'sum': np.zeros(3), 'sum_sq': np.zeros(3), 'count': 0})
        
        for idx in batch_indices:
            img, mask = self.dataset[idx]
            img_np = img.numpy().transpose(1, 2, 0)
            mask_np = mask.numpy()
            
            # 全局统计
            batch_rgb_sum += img_np.sum(axis=(0, 1))
            batch_rgb_sum_sq += (img_np**2).sum(axis=(0, 1))
            batch_pixels += img_np.size // 3
            
            # 分类统计
            for cls in np.unique(mask_np):
                if cls == 0: continue
                mask_cls = (mask_np == cls)
                rgb_values = img_np[mask_cls]
                class_stats[cls]['sum'] += rgb_values.sum(axis=0)
                class_stats[cls]['sum_sq'] += (rgb_values**2).sum(axis=0)
                class_stats[cls]['count'] += mask_cls.sum()
                
        return batch_rgb_sum, batch_rgb_sum_sq, batch_pixels, class_stats

    def parallel_analyze(self):
        pool = mp.Pool(self.num_workers)
        results = pool.map(self.compute_batch_stats, [list(batch) for batch in self])
        pool.close()
        
        # 聚合结果
        total_sum = np.zeros(3)
        total_sum_sq = np.zeros(3)
        total_pixels = 0
        total_class_stats = defaultdict(lambda: {'sum': np.zeros(3), 'sum_sq': np.zeros(3), 'count': 0})
        
        for b_sum, b_sum_sq, b_pixels, b_class in results:
            total_sum += b_sum
            total_sum_sq += b_sum_sq
            total_pixels += b_pixels
            for cls in b_class:
                for k in ['sum', 'sum_sq', 'count']:
                    total_class_stats[cls][k] += b_class[cls][k]
                    
        return total_sum, total_sum_sq, total_pixels, total_class_stats
    
def gpu_analyze(dataset, batch_size=256):
    device = torch.device('cuda')
    total_sum = torch.zeros(3, device=device)
    total_sum_sq = torch.zeros(3, device=device)
    total_pixels = 0
    class_stats = defaultdict(lambda: {
        'sum': torch.zeros(3, device=device),
        'sum_sq': torch.zeros(3, device=device),
        'count': torch.tensor(0, device=device)  # 改为tensor类型保持一致性
    })
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 全局统计
        total_sum += imgs.sum(dim=[0,2,3])
        total_sum_sq += (imgs**2).sum(dim=[0,2,3])
        total_pixels += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        
        # 分类统计
        for cls in valid_classes:
            mask = (masks == cls)
            count = mask.sum()
            if count == 0:
                continue
                
            rgb_values = imgs * mask.unsqueeze(1).float()
            class_stats[cls]['sum'] += rgb_values.sum(dim=[0,2,3])
            class_stats[cls]['sum_sq'] += (rgb_values**2).sum(dim=[0,2,3])
            class_stats[cls]['count'] += count
    
    # 转换为CPU numpy
    stats = {
        'total_mean': (total_sum / total_pixels).cpu().numpy(),
        'total_std': torch.sqrt(total_sum_sq/total_pixels - (total_sum/total_pixels)**2).cpu().numpy(),
        'class_stats': {
            cls: {
                'mean': (class_stats[cls]['sum']/class_stats[cls]['count']).cpu().numpy(),
                'std': torch.sqrt(
                    class_stats[cls]['sum_sq']/class_stats[cls]['count'] - 
                    (class_stats[cls]['sum']/class_stats[cls]['count'])**2
                ).cpu().numpy(),
                'count': int(class_stats[cls]['count'].cpu())  # 显式转换为int
            } for cls in class_stats if class_stats[cls]['count'] > 0
        }
    }
    return stats

def get_stats(dataset, batch_size=32):
    if torch.cuda.is_available():
        stats = gpu_analyze(dataset)
        # GPU结果转换
        total_mean = stats['total_mean']
        total_std = stats['total_std']
        
        # 确保所有类别都有统计值
        class_counts = {}
        rgb_means = {}
        rgb_stds = {}
        for cls in valid_classes:
            if cls in stats['class_stats']:
                class_counts[cls] = stats['class_stats'][cls]['count']
                rgb_means[cls] = stats['class_stats'][cls]['mean']
                rgb_stds[cls] = stats['class_stats'][cls]['std']
            else:
                class_counts[cls] = 0
                rgb_means[cls] = np.zeros(3)
                rgb_stds[cls] = np.zeros(3)
                
    else:
        # CPU多进程处理
        batch_sampler = StatsBatchSampler(dataset, batch_size=64,
                                        num_workers=0)
        total_sum, total_sum_sq, total_pixels, class_stats = batch_sampler.parallel_analyze()
        
        # CPU结果转换
        total_mean = total_sum / total_pixels
        total_std = np.sqrt(total_sum_sq/total_pixels - total_mean**2)
        
        class_counts = {}
        rgb_means = {}
        rgb_stds = {}
        for cls in valid_classes:
            if cls in class_stats and class_stats[cls]['count'] > 0:
                class_counts[cls] = int(class_stats[cls]['count'])
                rgb_means[cls] = class_stats[cls]['sum'] / class_stats[cls]['count']
                rgb_stds[cls] = np.sqrt(
                    class_stats[cls]['sum_sq']/class_stats[cls]['count'] - 
                    (class_stats[cls]['sum']/class_stats[cls]['count'])**2
                )
            else:
                class_counts[cls] = 0
                rgb_means[cls] = np.zeros(3)
                rgb_stds[cls] = np.zeros(3)
    
    return class_counts, rgb_means, rgb_stds, total_mean, total_std

if __name__ == '__main__':
    print("脚本开始执行...")
    torch.cuda.empty_cache()  # 清理残留的显存占用
    print(f"当前可用显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    print(f"BF16支持: {torch.cuda.is_bf16_supported()}")
    print(f"TF32支持: {torch.backends.cuda.matmul.allow_tf32}")  # RTX 40系列默认启用
    # 数据增强
    IMG_SIZE = 1024 // 2
    
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
    print(f"总样本数: {len(full_dataset)}")
    
    # 执行统计分析
    batch_size = 16
    class_counts, rgb_means, rgb_stds, total_mean, total_std = get_stats(full_dataset, batch_size)
    
    # 可视化
    plt.figure(figsize=(20, 6))
    plt.suptitle(f"混合数据集RGB通道分析 (IMG_SIZE={IMG_SIZE})", fontsize=14)
    
    # 1. 类别分布柱状图
    plt.subplot(1, 4, 1)
    sns.barplot(x=[class_names[k] for k in class_counts.keys()], 
                y=list(class_counts.values()),
                hue=[class_names[k] for k in class_counts.keys()],
                palette="viridis",
                legend=False)
    plt.title("类别像素分布\n(排除未标注)")
    plt.xticks(rotation=45)
    plt.ylabel("像素数量 (log scale)")
    plt.yscale('log')
    
    # 2. 全局RGB统计
    plt.subplot(1, 4, 2)
    channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']
    x_pos = np.arange(len(channels))
    
    # 均值
    plt.bar(x_pos - 0.15, total_mean, width=0.3, color=colors, alpha=0.7, label='均值')
    # 标准差
    plt.bar(x_pos + 0.15, total_std, width=0.3, color=colors, alpha=0.3, label='标准差')
    
    plt.xticks(x_pos, channels)
    plt.title("全局RGB统计")
    plt.ylabel("像素值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 各类别RGB均值对比
    plt.subplot(1, 4, 3)
    for i, color in enumerate(['r', 'g', 'b']):
        means = [rgb_means[cls][i] for cls in valid_classes]
        plt.plot(means, marker='o', color=color, label=channels[i])
        
    plt.xticks(range(len(valid_classes)), [class_names[cls] for cls in valid_classes])
    plt.title("各类别RGB均值")
    plt.ylabel("像素值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 各类别RGB标准差对比
    plt.subplot(1, 4, 4)
    for i, color in enumerate(['r', 'g', 'b']):
        stds = [rgb_stds[cls][i] for cls in valid_classes]
        plt.plot(stds, marker='s', color=color, linestyle='--', label=channels[i])
        
    plt.xticks(range(len(valid_classes)), [class_names[cls] for cls in valid_classes])
    plt.title("各类别RGB标准差")
    plt.ylabel("标准差")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rgb_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    print("\n[全局RGB统计]")
    print(f"均值(R/G/B): {total_mean.round(2)}")
    print(f"标准差(R/G/B): {total_std.round(2)}")
    
    print("\n[各类别RGB均值]")
    for cls in valid_classes:
        print(f"{class_names[cls]:<8}: {rgb_means[cls].round(2)}")
    
    print("\n[各类别RGB标准差]")
    for cls in valid_classes:
        print(f"{class_names[cls]:<8}: {rgb_stds[cls].round(2)}")















