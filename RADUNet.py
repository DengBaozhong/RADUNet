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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # 导入权重枚举
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold

random.seed(42)  # 固定Python随机种子
np.random.seed(42)  # 固定Numpy随机种子
torch.manual_seed(42)  # 固定PyTorch随机种子
torch.cuda.manual_seed_all(42)  # 固定所有CUDA随机种子
torch.backends.cudnn.deterministic = True  # 确保CuDNN结果可复现
torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动优化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 1024 // 2

# 更新后的类别映射
class_dict_LoveDA = {
    1: 5,   # 背景 -> 人工表面(包含道路和其他)
    2: 1,   # 建筑 -> 建筑区域
    3: 5,   # 道路 -> 人工表面(包含道路和其他)
    4: 3,   # 水体 -> 水域
    5: 4,   # 裸地 -> 裸地
    6: 2,   # 植被 -> 植被(包含农田)
    7: 2,   # 农田 -> 植被(包含农田)
    8: 5,   # Playground -> 人工表面(包含道路和其他)
    0: 0,   # 未标注 -> 未标注
}

class_dict_OpenEarthMap = {
    1: 4,   # Bareland -> 裸地与未利用地
    2: 2,   # rangeland -> 植被(包含农田)
    3: 5,   # developed space -> 人工表面(包含道路和其他)
    4: 5,   # Road -> 人工表面(包含道路和其他)
    5: 2,   # Tree -> 植被(包含农田)
    6: 3,   # Water -> 水域
    7: 2,   # Agriculture -> 植被(包含农田)
    8: 1,   # Building -> 建筑区域
    0: 0,   # 未标注 -> 未标注
}

# 自定义数据集类
class LoveDADataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.resize = transforms.Resize(size=(IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(size=(IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST)
        
        # 获取图像和掩码文件名列表
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        # 确保图像和掩码文件一一对应
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不匹配"
        for img, mask in zip(self.image_files, self.mask_files):
            assert img.replace('.png', '') == mask.replace('.png', ''), f"图像和掩码不匹配: {img} vs {mask}"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        try:
            # 使用PIL读取图像和掩码
            image = Image.open(img_path).convert('RGB')  # 确保是3通道
            mask = Image.open(mask_path)
            # 统一resize
            image = self.resize(image)
            mask = self.resize_mask(mask)
            # 转换为numpy数组
            image = np.array(image).astype(np.float32)
            mask = np.array(mask)
            
            # 将掩码转换为我们的类别
            mask = self.map_classes(mask)
            
            if self.transform:
                # print("应用数据增强...")
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                # 确保掩码是Long类型
                mask = mask.long()  # 添加这行
            else:
                # 如果没有transform，至少需要将图像归一化并转为tensor
                image = transforms.functional.to_tensor(image)
                mask = torch.from_numpy(mask).long()
            return image, mask
    
        except Exception as e:
            print(f"加载第 {idx} 个样本时出错: {str(e)}")
            raise

    def map_classes(self, mask):
        # 将原始类别映射到我们的大类
        new_mask = np.zeros_like(mask)
        for original_class, target_class in class_dict_LoveDA.items():
            new_mask[mask == original_class] = target_class
        return new_mask

# 自定义OpenEarthMap数据集类
class OpenEarthMapDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.resize = transforms.Resize(size=(IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(size=(IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST)
        
        # 获取图像和掩码文件名列表
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
        # 确保图像和掩码文件一一对应
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不匹配"
        for img, mask in zip(self.image_files, self.mask_files):
            assert img.replace('.png', '') == mask.replace('.png', ''), f"图像和掩码不匹配: {img} vs {mask}"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        try:
            # 使用PIL读取图像和掩码
            image = Image.open(img_path).convert('RGB')  # 确保是3通道
            mask = Image.open(mask_path)
            # 统一resize
            image = self.resize(image)
            mask = self.resize_mask(mask)
            # 转换为numpy数组
            image = np.array(image).astype(np.float32)
            mask = np.array(mask)
            
            # 将掩码转换为我们的类别
            mask = self.map_classes(mask)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                # 确保掩码是Long类型
                mask = mask.long()
            else:
                # 如果没有transform，至少需要将图像归一化并转为tensor
                image = transforms.functional.to_tensor(image)
                mask = torch.from_numpy(mask).long()
            return image, mask
    
        except Exception as e:
            print(f"加载第 {idx} 个样本时出错: {str(e)}")
            raise

    def map_classes(self, mask):
        # 将OpenEarthMap原始类别映射到我们的大类
        new_mask = np.zeros_like(mask)
        for original_class, target_class in class_dict_OpenEarthMap.items():
            new_mask[mask == original_class] = target_class
        return new_mask

class AttentionBlock(nn.Module):
    """简单的注意力模块（CBAM风格）"""
    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel = self.channel_attention(x) * x
        # 空间注意力（沿通道维度取均值和最大值）
        spatial_mean = torch.mean(channel, dim=1, keepdim=True)
        spatial_max = torch.max(channel, dim=1, keepdim=True)[0]
        spatial = torch.cat([spatial_mean, spatial_max], dim=1)
        spatial = self.spatial_attention(spatial) * channel
        return spatial

class ResNetUNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        
        base_channels = 64
        
        # 加载预训练 ResNet
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # 使用ResNet50
        
        # 编码器部分（提取 ResNet 的各个阶段）
        self.encoder = nn.ModuleDict({
            "conv1": nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu),
            "maxpool": base_model.maxpool,
            "layer1": base_model.layer1,  # [256, H/4, W/4] (ResNet50的输出通道数不同)
            "layer2": base_model.layer2,  # [512, H/8, W/8]
            "layer3": base_model.layer3,  # [1024, H/16, W/16]
            "layer4": base_model.layer4   # [2048, H/32, W/32]
        })
        
        # 解码器部分（自定义）
        self.attn1 = AttentionBlock(1024)  # 对应layer3的输出通道(1024)
        self.attn2 = AttentionBlock(512)   # 对应layer2的输出通道(512)
        self.attn3 = AttentionBlock(256)   # 对应layer1的输出通道(256)

        self.up1 = Up(2048 + 1024, 1024)  # 输入通道数需匹配 skip connection
        self.up2 = Up(1024 + 512, 512)
        self.up3 = Up(512 + 256, 256)
        self.up4 = Up(256 + 64, base_channels)     # 额外处理初始的 conv1 输出
        
        # 添加深度监督
        self.side_out1 = nn.Conv2d(2048, n_classes, kernel_size=1)
        self.side_out2 = nn.Conv2d(1024, n_classes, kernel_size=1)

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # 编码器部分
        x0 = self.encoder["conv1"](x)       # [64, H/2, W/2]
        x1 = self.encoder["maxpool"](x0)   # [64, H/4, W/4]
        x1 = self.encoder["layer1"](x1)     # [64, H/4, W/4]
        x2 = self.encoder["layer2"](x1)    # [128, H/8, W/8]
        x3 = self.encoder["layer3"](x2)    # [256, H/16, W/16]
        x4 = self.encoder["layer4"](x3)    # [512, H/32, W/32]
        
        # 深度监督分支（在解码前计算辅助输出）
        side1 = self.side_out1(x4)          # [n_classes, H/32, W/32]
        side2 = self.side_out2(x3)          # [n_classes, H/16, W/16]

        # 解码器部分（注意跳跃连接的拼接）
        x = self.up1(x4, self.attn1(x3))          # [256, H/16, W/16]
        x = self.up2(x, self.attn2(x2))           # [128, H/8, W/8]
        x = self.up3(x, self.attn3(x1))           # [64, H/4, W/4]
        x = self.up4(x, x0)           # [32, H/2, W/2]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # 恢复到原图尺寸
        main_out = self.outc(x)             # [n_classes, H, W]
        
        # 对辅助输出上采样到原图尺寸
        side1 = F.interpolate(side1, scale_factor=32, mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2, scale_factor=16, mode='bilinear', align_corners=True)

        return main_out, side1, side2

# 定义U-Net模型（保持不变）
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """支持跳跃连接的 Upsample 模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 拼接 skip connection
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean', ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index  # 新增参数：忽略的类别索引

    def forward(self, inputs, targets):
        # 创建掩码：忽略targets中值为ignore_index的像素
        mask = (targets != self.ignore_index)
        masked_targets = targets[mask]
        masked_inputs = inputs[mask.unsqueeze(1).expand_as(inputs)].view(-1, inputs.shape[1])
        
        ce_loss = F.cross_entropy(masked_inputs, masked_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index  # 新增参数
    
    def forward(self, pred, target):
        pred = pred.softmax(dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
        
        # 创建掩码：忽略target中值为ignore_index的像素
        mask = (target != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
        pred = pred * mask
        target_onehot = target_onehot * mask
        
        # 向量化计算加速
        intersection = torch.sum(pred * target_onehot, dim=(2,3))
        union = torch.sum(pred, dim=(2,3)) + torch.sum(target_onehot, dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=3, ignore_index=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index  # 新增参数：忽略的类别索引
        self.edge_kernel = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.edge_kernel.weight.data = self._get_sobel_kernel(kernel_size)
        self.edge_kernel.requires_grad_(False)

    def _get_sobel_kernel(self, ksize):
        kernel = torch.ones(1, 1, ksize, ksize).to(device) / (ksize**2 - 1)
        kernel[0, 0, ksize//2, ksize//2] = -1
        return kernel

    def forward(self, pred, target):
        with torch.no_grad():
            # 生成onehot标签 [B, C, H, W]
            target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            target_edges = torch.zeros_like(target_onehot)
            
            # 逐类别计算边缘（跳过ignore_index）
            for c in range(target_onehot.shape[1]):
                if c == self.ignore_index:  # 跳过无效类别
                    continue
                edge = self.edge_kernel(target_onehot[:, c:c+1])  # 保持4D输入 [B, 1, H, W]
                target_edges[:, c] = edge.squeeze(1)  # 移除通道维度 [B, H, W]

            # 取所有类别的最大边缘响应（排除无效类别）
            valid_classes = [i for i in range(target_onehot.shape[1]) if i != self.ignore_index]
            target_edges = target_edges[:, valid_classes].max(dim=1)[0]  # [B, H, W]

        # 计算预测边缘（同样跳过ignore_index）
        pred_softmax = pred.softmax(dim=1)
        pred_edges = torch.zeros_like(pred_softmax)
        for c in range(pred_softmax.shape[1]):
            if c == self.ignore_index:  # 跳过无效类别
                continue
            edge = self.edge_kernel(pred_softmax[:, c:c+1])
            pred_edges[:, c] = edge.squeeze(1)
        
        # 取有效类别的最大边缘响应
        pred_edges = pred_edges[:, valid_classes].max(dim=1)[0]  # [B, H, W]
        return F.mse_loss(pred_edges, target_edges)

def calculate_miou(outputs, masks, num_classes, ignore_index=0):
    """计算mIoU（忽略特定类别）"""
    preds = outputs.argmax(dim=1)  # [B, H, W]
    mask = (masks != ignore_index)  # [B, H, W]
    
    # 应用mask并展平
    preds = preds[mask]  # [N]
    masks = masks[mask]  # [N]
    
    if masks.numel() == 0:
        return 0.0
    
    # 向量化计算混淆矩阵（替代逐像素循环）
    conf_matrix = torch.zeros(num_classes, num_classes, device=outputs.device)
    indices = masks * num_classes + preds  # 将 (t, p) 映射到一维索引
    counts = torch.bincount(indices, minlength=num_classes**2)
    conf_matrix = counts.reshape(num_classes, num_classes)
    
    # 计算IoU
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# 定义训练函数
def train_model(model, train_loader, val_loader, focal_loss=None, dice_loss=None, boundary_loss=None, optimizer=None, num_epochs=25, fold=0):
    print("\n开始训练过程...")
    best_loss = float('inf')
    best_miou = 0.0
    tolerance = 1e-4
    scaler = GradScaler()  # 新增梯度缩放器
    
    # 初始化默认损失函数（如果未提供）
    if focal_loss is None:
        focal_loss = FocalLoss(alpha=0.75, gamma=2).to(device)
    if dice_loss is None:
        dice_loss = DiceLoss().to(device)
    if boundary_loss is None:
        boundary_loss = BoundaryLoss().to(device)
    
    # 早停参数
    patience = 10  # 允许验证损失不下降的轮数
    no_improve = 0  # 记录连续未改进的轮数
    
    for epoch in range(num_epochs):
        print(f"\n开始第 {epoch+1}/{num_epochs} 轮训练...")
        model.train()
        running_loss = 0.0
        
        # 训练阶段
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
    
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 用autocast包裹前向计算
            with autocast(device_type='cuda', 
                       dtype=torch.bfloat16):
                main_out, side1, side2 = model(images)
                
                # 计算各损失分量
                fl = focal_loss(main_out, masks)
                dl = dice_loss(main_out, masks)
                
                # 计算辅助输出损失（权重可调整）
                fl_side1 = focal_loss(side1, masks) * 0.5  # 辅助损失权重设为0.5
                fl_side2 = focal_loss(side2, masks) * 0.3
                dl_side1 = dice_loss(side1, masks) * 0.5
                dl_side2 = dice_loss(side2, masks) * 0.3

                if epoch >= 5:  # 前5个epoch不使用边界损失
                    bl = boundary_loss(main_out, masks)
                    loss = 0.5*fl + 0.3*dl + 0.2*bl + (fl_side1 + fl_side2 + dl_side1 + dl_side2)
                    # 数值稳定性检查
                    if torch.isnan(loss):
                        print(f"NaN detected! Components - Focal: {fl.item()}, Dice: {dl.item()}, Boundary: {bl.item()}")
                        raise RuntimeError("Training terminated due to NaN loss")
                else:
                    loss = 0.7*fl + 0.3*dl + (fl_side1 + fl_side2 + dl_side1 + dl_side2)
                    # 数值稳定性检查
                    if torch.isnan(loss):
                        print(f"NaN detected! Components - Focal: {fl.item()}, Dice: {dl.item()}")
                        raise RuntimeError("Training terminated due to NaN loss")
                
            # 替换原来的loss.backward()
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.unscale_(optimizer)     # 必须先unscale！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
            scaler.step(optimizer)         # 缩放后的优化器步进
            scaler.update()                # 更新缩放因子
            
            running_loss += loss.item() * images.size(0)
            
            # 打印前几个批次的损失
            if batch_idx < 3:
                print(f"批次 {batch_idx} Train Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        print(f"\n开始第 {epoch+1}/{num_epochs} 轮验证...")
        total_loss = 0.0
        total_miou = 0.0
        model.eval()
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                main_out, _, _ = model(images)
                
                fl = focal_loss(main_out, masks)
                dl = dice_loss(main_out, masks)
                
                if epoch >= 5:  # 前5个epoch不使用边界损失
                    bl = boundary_loss(main_out, masks)
                    loss = 0.5*fl + 0.3*dl + 0.2*bl
                    # 数值稳定性检查
                    if torch.isnan(loss):
                        print(f"NaN detected! Components - Focal: {fl.item()}, Dice: {dl.item()}, Boundary: {bl.item()}")
                        raise RuntimeError("Training terminated due to NaN loss")
                else:
                    loss = 0.7*fl + 0.3*dl
                    # 数值稳定性检查
                    if torch.isnan(loss):
                        print(f"NaN detected! Components - Focal: {fl.item()}, Dice: {dl.item()}")
                        raise RuntimeError("Training terminated due to NaN loss")
                
                # 计算mIoU
                # batch_miou = calculate_miou(outputs, masks, n_classes)
                
                # 累积各项损失
                total_loss += loss.item() * images.size(0)
                # total_miou += batch_miou * images.size(0)

            # 计算epoch平均损失
            epoch_val_loss  = total_loss / len(val_loader.dataset)
            # epoch_miou = total_miou / len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  # f'Val mIoU: {epoch_miou:.4f}'
                  )
            
            # 保存最佳模型
            if epoch_val_loss < best_loss - tolerance:
                best_loss = epoch_val_loss
                no_improve = 0  # 重置计数器
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')  # 文件名只包含fold号
                print(f"发现更好的模型，已保存为 best_model_fold{fold+1}.pth!")
            # if epoch_miou > best_miou + tolerance:  # 改为监控mIoU提升
            #     best_miou = epoch_miou
            #     no_improve = 0
            #     torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')  # 文件名只包含fold号
            #     print(f"发现更好的模型，已保存为 best_model_fold{fold+1}.pth!")
            else:
                no_improve += 1  # 验证损失未下降，计数器+1
                print(f"验证损失未改进，连续 {no_improve}/{patience} 轮")
                # 检查是否触发早停
                if no_improve >= patience:
                    print(f"早停触发！连续 {patience} 轮验证损失未下降。")
                    break  # 终止训练
    
    print('Training complete')
    return model 
    
if __name__ == '__main__':
    print("脚本开始执行...")
    torch.cuda.empty_cache()  # 清理残留的显存占用
    print(f"当前可用显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    print(f"BF16支持: {torch.cuda.is_bf16_supported()}")
    print(f"TF32支持: {torch.backends.cuda.matmul.allow_tf32}")  # RTX 40系列默认启用
    # 数据增强
    IMG_SIZE = 1024 // 2
    train_transform = A.Compose([
        # 几何变换（保持地理空间关系）
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.5, 1.0),  # 缩放范围（相对于原始图像）
            interpolation=1,    # 插值方法（1=双线性）
            p=0.8               # 执行概率
        ),
        A.Rotate(limit=45, p=0.8, border_mode=0),  # 旋转+边缘填充
        A.RandomRotate90(),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # 弹性变形模拟地形起伏
        
        # 辐射变换（模拟不同成像条件）
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # 增强局部对比度
        A.RandomShadow(shadow_roi=(0, 0, 1, 0.5), shadow_dimension=5, p=0.2),  # 模拟云层阴影

        # 空间扭曲（增强鲁棒性）
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.OpticalDistortion(distort_limit=0.2, p=0.3),
        
        # 噪声和模糊（模拟传感器差异）
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        
        # 高级增强（针对地图特性）
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.1),  # 太阳耀斑
        A.RandomFog(p=0.1),  # 雾霾效果
        
        # 标准化和Tensor转换
        A.Resize(IMG_SIZE, IMG_SIZE),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        A.Normalize(mean=[0.3481, 0.3759, 0.3497], std=[0.1875, 0.1689, 0.1627], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        A.Normalize(mean=[0.3481, 0.3759, 0.3497], std=[0.1875, 0.1689, 0.1627], max_pixel_value=255.0),
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
            
            # 筛选出属于当前子数据集的train_ids和val_ids
            sub_train_ids = [i - start_idx for i in train_ids if start_idx <= i < end_idx]
            sub_val_ids = [i - start_idx for i in val_ids if start_idx <= i < end_idx]
            
            # 创建子数据集的Subset
            train_subset = torch.utils.data.Subset(dataset, sub_train_ids)
            val_subset = torch.utils.data.Subset(dataset, sub_val_ids)
            
            # 设置transform
            train_subset.dataset.transform = train_transform
            val_subset.dataset.transform = val_transform
            
            train_subsets.append(train_subset)
            val_subsets.append(val_subset)
        
        # 合并所有子数据集的Subset
        train_dataset = torch.utils.data.ConcatDataset(train_subsets)
        val_dataset = torch.utils.data.ConcatDataset(val_subsets)
        
        # 创建数据加载器
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)
        
        # 检查是否存在之前训练过的模型
        model_path = f'best_model_fold{fold+1}.pth'
        n_channels = 3  # RGB图像
        n_classes = 6   # 我们的6个类别
        if os.path.exists(model_path):
            print(f"发现之前训练的模型 {model_path}，加载模型继续训练...")
            model = ResNetUNet(n_classes=n_classes, pretrained=False).to(device)  # 不加载预训练权重
            model.load_state_dict(torch.load(model_path))
            print("模型加载成功，继续训练...")
        else:
            print(f"未找到之前训练的模型 {model_path}，初始化新模型...")
            model = ResNetUNet(n_classes=n_classes, pretrained=True).to(device)  # 加载预训练权重
        
        focal_loss = FocalLoss(alpha=0.75, gamma=2).to(device)
        dice_loss = DiceLoss().to(device)
        boundary_loss = BoundaryLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        
        # 训练模型
        trained_model = train_model(model, train_loader, val_loader, 
                                    focal_loss=focal_loss,dice_loss=dice_loss,boundary_loss=boundary_loss, 
                                    optimizer=optimizer, num_epochs=100, fold=fold)
        
        torch.cuda.empty_cache()  # 清理残留的显存占用

    # 所有折完成后，计算平均性能指标
    print("\n=== 交叉验证完成 ===")
    torch.cuda.empty_cache()  # 清理残留的显存占用