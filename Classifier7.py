import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from H5PoseDataset import H5PoseDataset
from ActionClassifier_GPT import PoseGCNTransformer
from Loss_GCN import CombinedLoss


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_and_validate(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(config['seed'])

    # 创建优化后的模型
    model = PoseGCNTransformer(
        num_joints=config['num_joints'],
        in_channels=3,
        hidden_dim=256,  # 更高的隐藏维度
        num_classes=config['num_classes'],
        num_gcn_layers=3,  # 增加GCN层数
        num_transformer_layers=6,  # 更多Transformer层
        feat_dim=128,  # 特征维度
        num_heads=16  # Transformer头数
    ).to(device)
    model = DDP(model, device_ids=[device])

    # 创建数据集（增强策略优化）
    train_dataset = H5PoseDataset(
        h5_path=config['data_path'],
        group='train',
        segment_length=config['segment_length'],
        augment=True,
        reverse_prob=0.8,  # 更高的倒序概率
        mask_ratio=0.2,  # 时间掩码比例
        noise_scale=0.02  # 增加噪声强度
    )

    val_dataset = H5PoseDataset(
        h5_path=config['data_path'],
        group='validation',
        segment_length=config['segment_length'],
        augment=False
    )

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # 数据加载器（增大批次大小）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 损失函数与优化器（调整权重和优化器）
    criterion = CombinedLoss(
        classification_weight=0.8,  # 分类损失为主
        contrastive_weight=0.2,  # 降低对比损失权重
        margin=0.8  # 更小的对比边界
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01  # 权重衰减
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # 余弦退火周期
        eta_min=1e-5
    )

    scaler = GradScaler()

    if rank == 0:
        writer = SummaryWriter(log_dir=config['log_dir'])
        best_val_acc = 0.0

    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast():
                logits, features = model(data)

                # 简化对比对生成，避免NaN（使用相邻样本作为对比对）
                B = data.size(0)
                contrast_pairs = [(i, (i + 1) % B) for i in range(B)]  # 相邻样本对
                loss = criterion(logits, features, target, contrast_pairs)

            # 梯度裁剪防止爆炸
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 统计指标（CPU转换用于安全打印）
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted.cpu() == target.cpu()).sum().item()
            total += target.size(0)

            # 打印进度
            if rank == 0 and batch_idx % config['log_interval'] == 0:
                print(f"Rank {rank} | Epoch {epoch + 1}/{config['num_epochs']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, features = model(data)

                B = data.size(0)
                constrast_pairs = [(i, (i+1)%B) for i in range(B)]

                loss = criterion(logits, features, target, constrast_pairs)  # 验证阶段可只计算分类损失

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += (predicted.cpu() == target.cpu()).sum().item()
                val_total += target.size(0)

        # 跨进程同步指标
        train_loss = torch.tensor(total_loss / len(train_loader), device=device)
        train_acc = torch.tensor(correct / total, device=device)
        val_loss_tensor = torch.tensor(val_loss / len(val_loader), device=device)
        val_acc = torch.tensor(val_correct / val_total, device=device)

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)

        train_loss /= world_size
        train_acc /= world_size
        val_loss = val_loss_tensor.item() / world_size
        val_acc /= world_size

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), config['save_path'])
                print(f"Saved best model with accuracy: {best_val_acc * 100:.2f}%")

    if rank == 0:
        writer.close()
    cleanup()


if __name__ == "__main__":
    # 配置参数（关键优化点）
    config = {
        'data_path': 'split_data.h5',  # H5数据路径
        'num_joints': 133,  # 关节数量
        'num_classes': 2,  # 分类类别数
        'segment_length': 300,  # 序列长度
        'batch_size': 32,  # 批次大小（增大至32）
        'num_epochs': 200,  # 训练轮数（延长至200）
        'num_workers': 8,  # 数据加载线程数
        'log_interval': 50,  # 日志间隔
        'seed': 42,  # 随机种子
        'log_dir': 'runs/action_classifier_v2',  # 日志目录
        'save_path': 'checkpoints/best_model.pth',  # 模型保存路径
        'reverse_prob': 0.8,  # 倒序概率
        'mask_ratio': 0.2,  # 时间掩码比例
        'noise_scale': 0.02,  # 噪声强度
    }

    # 创建保存目录
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)

    # 启动分布式训练
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(
            train_and_validate,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # 单卡训练（若没有GPU）
        train_and_validate(-1, 1, config)
