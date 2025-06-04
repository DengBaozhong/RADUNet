# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 00:18:28 2025

@author: maxim
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
import random
import os
import matplotlib.pyplot as plt
from rtree import index
import glob

# 使用系统字体（推荐）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def stratified_random_sampling(
    province_geojson_path,
    num_samples=10,
    square_size=4096,
    max_attempts_multiplier=5
):
    """
    分层随机采样（Stratified Random Sampling）改进版
    
    参数:
    - province_geojson_path: 省份GeoJSON路径
    - num_samples: 所需样本数
    - square_size: 采样方块边长(米)
    - max_attempts_multiplier: 最大尝试倍数（网格密度系数）
    """
    try:
        # 加载数据并转换坐标系
        province = gpd.read_file(province_geojson_path)
        if not province.crs.is_projected:
            province = province.to_crs(epsg=3857)
        
        # 修复无效几何图形
        province.geometry = province.geometry.apply(lambda geom: make_valid(geom)) if not province.geometry.is_valid.all() else province.geometry
        
        # 准备可视化
        fig, ax = plt.subplots(figsize=(12, 12))
        province.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        
        bounds = province.total_bounds
        province_name = os.path.splitext(os.path.basename(province_geojson_path))[0]
        
        # 创建省份特定的输出文件夹
        output_dir = f"output_samples_{province_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 初始化空间索引
        idx = index.Index()
        samples = []
        
        # 动态计算网格划分层数
        min_grid_size = int(np.sqrt(num_samples))  # 最小网格数
        grid_size = min_grid_size
        total_attempts = 0
        generated_samples = 0
        
        # 渐进式增加网格密度直到满足需求
        while generated_samples < num_samples and grid_size <= min_grid_size * max_attempts_multiplier:
            # 计算网格步长
            x_step = (bounds[2] - bounds[0]) / grid_size
            y_step = (bounds[3] - bounds[1]) / grid_size
            
            # 遍历所有网格单元
            for i in range(grid_size):
                for j in range(grid_size):
                    total_attempts += 1
                    
                    # 在当前网格内随机生成中心点
                    center_x = bounds[0] + (i + random.random()) * x_step
                    center_y = bounds[1] + (j + random.random()) * y_step
                    
                    # 创建方形
                    half_size = square_size / 2
                    square = box(
                        center_x - half_size, center_y - half_size,
                        center_x + half_size, center_y + half_size
                    )
                    
                    # 检查有效性 - 使用更稳健的方法
                    try:
                        if not any(geom.contains(square) for geom in province.geometry):
                            continue
                    except:
                        continue
                    
                    # 检查重叠
                    square_bounds = square.bounds
                    nearby = list(idx.intersection(square_bounds))
                    if any(square.intersects(samples[k]) for k in nearby):
                        continue
                    
                    # 合格样本处理
                    samples.append(square)
                    idx.insert(len(samples)-1, square_bounds)
                    sample_num = len(samples)  # 当前样本编号
                    
                    # 可视化
                    gpd.GeoSeries(square).plot(
                        ax=ax, edgecolor='red', facecolor='none', 
                        linewidth=2, label=f'Sample {sample_num}'
                    )
                    
                    # 添加文本标注 - 新增加这部分
                    centroid = square.centroid
                    ax.text(
                        centroid.x, centroid.y, f's{sample_num}', 
                        fontsize=6, color='black', ha='center', va='center',
                        # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                    )

                    # 保存GeoJSON
                    sample_gdf = gpd.GeoDataFrame(geometry=[square], crs=province.crs)
                    sample_gdf = sample_gdf.to_crs(epsg=4326)
                    output_path = f"{output_dir}/{province_name}_stratified_s{len(samples)}.geojson"
                    sample_gdf.to_file(output_path, driver='GeoJSON')
                    
                    generated_samples += 1
                    if generated_samples >= num_samples:
                        break
                if generated_samples >= num_samples:
                    break
            
            # 增加网格密度
            grid_size += 1
        
        # 可视化增强
        ax.set_title(
            f"{province_name} stratified random sampling\n"
            f"Successful samples: {generated_samples}/{num_samples} | "
            f"Grid density: {grid_size-1}x{grid_size-1} | "
            f"Total attempts: {total_attempts}",
            fontsize=14
        )
        
        # 添加网格线（可视化辅助）
        for i in range(grid_size):
            ax.axvline(bounds[0] + i*x_step, color='blue', alpha=0.2, linestyle='--', linewidth=0.5)
            ax.axhline(bounds[1] + i*y_step, color='blue', alpha=0.2, linestyle='--', linewidth=0.5)
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{province_name}_stratified_samples.png", dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以避免内存泄漏

        print(f"► {province_name} 成功生成 {generated_samples}/{num_samples} 个样本")
        print(f"► 最终网格密度: {grid_size-1}x{grid_size-1}")
        print(f"► 可视化保存至: {output_dir}/{province_name}_stratified_samples.png")
    
    except Exception as e:
        print(f"处理 {province_name} 时出错: {str(e)}")
        if 'fig' in locals():
            plt.close()

# 使用示例
if __name__ == "__main__":
    # 获取当前目录下所有省份JSON文件
    province_files = glob.glob("*.json")
    num_samples = 100
    
    if not province_files:
        print("未找到任何省份JSON文件！请确保文件格式为'省份名.json'")
    else:
        for province_file in province_files:
            print(f"\n正在处理省份: {province_file}")
            province_name = os.path.splitext(os.path.basename(province_file))[0]
            # 特殊处理澳门和香港
            if "Macau" in province_name:
                # 减小采样方块尺寸以适应小面积地区
                stratified_random_sampling(province_file, num_samples=num_samples, square_size=512)
            elif "Hong kong" in province_name:
                stratified_random_sampling(province_file, num_samples=num_samples, square_size=1024)
            elif "Shanghai" in province_name:
                stratified_random_sampling(province_file, num_samples=num_samples, square_size=6144)
            else:
                stratified_random_sampling(province_file, num_samples=num_samples, square_size=8192)
