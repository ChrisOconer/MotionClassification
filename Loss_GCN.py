import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """加权焦损失（保留原功能，供CombinedLoss调用）"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class ContrastiveLoss(nn.Module):
    """对比损失：计算两个特征向量的欧氏距离对比损失"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feature1, feature2, label):
        """
        参数:
            feature1: 特征向量1 [B, D]
            feature2: 特征向量2 [B, D]
            label: 对比标签 (1=相似, 0=不相似) [B]
        """
        dist = torch.norm(feature1 - feature2, p=2, dim=1)
        loss = torch.mean(
            label * torch.pow(dist, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss

class CombinedLoss(nn.Module):
    """组合损失：分类损失 + 对比损失"""
    def __init__(
        self,
        classification_weight=0.7,  # 分类损失权重
        contrastive_weight=0.3,     # 对比损失权重
        margin=1.0                  # 对比损失边界
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.classification_loss = WeightedFocalLoss()  # 可替换为CrossEntropyLoss
        self.contrastive_loss = ContrastiveLoss(margin=margin)

    def forward(
        self,
        logits,          # 分类预测输出 [B, C]
        features,        # 特征向量 [B, D]
        targets,         # 分类标签 [B]
        contrast_pairs=None  # 对比样本对 [(feature1_idx, feature2_idx), ...]
    ):
        """
        参数:
            contrast_pairs: 对比样本对列表，默认使用全样本对比
        """
        # 1. 计算分类损失
        class_loss = self.classification_loss(logits, targets)

        # 2. 计算对比损失
        if features is None:
            contrast_loss = 0
        else:
            if contrast_pairs is None:
                # 生成全样本对比对（同类为正样本，不同类为负样本）
                B = features.size(0)
                labels = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()  # 正样本对标签矩阵 [B, B]
                labels = labels.view(-1)  # 展平为 [B*B]
                features1 = features.repeat(B, 1)  # [B*B, D]
                features2 = features.repeat(1, B).view(-1, features.size(1))  # [B*B, D]
                contrast_loss = self.contrastive_loss(features1, features2, labels)
            else:
                # 使用自定义对比对
                feature_pairs = torch.stack([features[i] for i, j in contrast_pairs] + [features[j] for i, j in contrast_pairs])
                feature_pairs = feature_pairs.view(-1, 2, features.size(1))
                labels = torch.tensor([1 if i == j else 0 for (i, j) in contrast_pairs], device=features.device)
                contrast_loss = self.contrastive_loss(feature_pairs[:, 0], feature_pairs[:, 1], labels)

        # 3. 组合损失
        total_loss = (
            self.classification_weight * class_loss +
            self.contrastive_weight * contrast_loss
        )
        return total_loss
