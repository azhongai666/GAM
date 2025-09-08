#!/usr/bin/env python3
"""
GAM (Gradient norm Aware Minimization) 使用示例
基于README中的示例代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse

# 导入GAM相关模块
from gam.gam import GAM
from gam.smooth_cross_entropy import smooth_crossentropy

class SimpleModel(nn.Module):
    """简单的神经网络模型用于演示"""
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_dummy_data(num_samples=1000, input_size=10, num_classes=2):
    """创建虚拟数据用于演示"""
    # 生成随机输入数据
    X = torch.randn(num_samples, input_size)
    # 生成随机标签
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return dataset

def main():
    # 创建参数对象，模拟命令行参数
    class Args:
        def __init__(self):
            # GAM相关参数（使用README中提到的默认值）
            self.grad_rho = 0.05        # rho' 参数
            self.grad_norm_rho = 0.1    # rho_t 参数  
            self.grad_beta_0 = 0.9      # beta 参数
            self.grad_beta_1 = 0.1      # alpha 参数
            self.grad_beta_2 = 1.0      # 辅助参数
            self.grad_beta_3 = 1.0      # 辅助参数
            self.grad_gamma = 0.1       # gamma 参数
            self.lr = 0.01
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.label_smoothing = 0.1
    
    args = Args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleModel().to(device)
    print("模型结构:")
    print(model)
    
    # 创建虚拟数据
    dataset = create_dummy_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 按照README示例初始化优化器
    print("\n=== 初始化GAM优化器 ===")
    
    # 1. 初始化基础优化器 (如SGD, Adam, AdamW ...)
    base_optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    print(f"基础优化器: {base_optimizer}")
    
    # 2. 创建学习率调度器（GAM需要）
    from gam.util import ProportionScheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=200)
    
    # 3. 创建GAM所需的调度器
    grad_rho_scheduler = ProportionScheduler(
        pytorch_lr_scheduler=lr_scheduler, 
        max_lr=args.lr, 
        min_lr=0.0,
        max_value=args.grad_rho, 
        min_value=args.grad_rho
    )
    
    grad_norm_rho_scheduler = ProportionScheduler(
        pytorch_lr_scheduler=lr_scheduler, 
        max_lr=args.lr, 
        min_lr=0.0,
        max_value=args.grad_norm_rho, 
        min_value=args.grad_norm_rho
    )
    
    # 4. 基于基础优化器初始化GAM优化器
    gam_optimizer = GAM(
        params=model.parameters(), 
        base_optimizer=base_optimizer, 
        model=model,
        grad_rho_scheduler=grad_rho_scheduler,
        grad_norm_rho_scheduler=grad_norm_rho_scheduler,
        args=args
    )
    print(f"GAM优化器: {gam_optimizer}")
    
    # 定义损失函数
    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()
    
    print("\n=== 开始训练演示 ===")
    model.train()
    
    # 训练几个batch来演示GAM的使用
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= 3:  # 只演示3个batch
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        
        # 按照README中的步骤使用GAM
        
        # GAM 设置 closure 并自动运行 predictions = model(inputs), loss = loss_fn(predictions, targets), loss.backward()
        gam_optimizer.set_closure(loss_fn, inputs, targets)
        
        # 更新模型参数
        predictions, loss = gam_optimizer.step()
        
        print(f"  预测形状: {predictions.shape}")
        print(f"  损失值: {loss.item():.4f}")
    
    print("\n=== GAM演示完成 ===")
    print("✓ 成功初始化GAM优化器")
    print("✓ 成功使用set_closure方法设置损失函数")
    print("✓ 成功执行GAM优化步骤")
    print("\n这展示了README中'How to use GAM'部分的核心用法！")

if __name__ == "__main__":
    main()