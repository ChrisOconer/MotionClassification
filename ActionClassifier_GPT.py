import torch
import torch.nn as nn
import torch.nn.functional as F



def get_adjacency_matrix(num_joints=133, normalize=True):
    """
    构建人体关节点邻接矩阵，表示关节间的连接关系

    参数:
        num_joints: 关节点数量
        normalize: 是否归一化邻接矩阵

    返回:
        adj_matrix: 邻接矩阵 [V, V]
    """
    # 定义人体关节连接关系
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 19), (17, 19), (18, 19), (16, 22), (20, 22), (21, 22),
        (91, 92), (92, 93), (93, 94), (94, 95),
        (96, 97), (97, 98), (98, 99), (100, 101), (101, 102), (102, 103),
        (104, 105), (105, 106), (106, 107), (108, 109), (109, 110), (110, 111),
        (112, 113), (113, 114), (114, 115), (115, 116),
        (117, 118), (118, 119), (119, 120), (121, 122), (122, 123), (123, 124),
        (125, 126), (126, 127), (127, 128), (129, 1303), (130, 131), (131, 132)
    ]

    # 构建邻接矩阵
    adj_matrix = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # 无向图

    # 自连接
    for i in range(num_joints):
        adj_matrix[i, i] = 1

    # 归一化
    if normalize:
        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        D = torch.diag(1.0 / torch.sqrt(adj_matrix.sum(dim=1) + 1e-6))
        adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)

    return adj_matrix


class SimpleGCNLayer(nn.Module):
    """简单图卷积网络层"""
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.adj_matrix = adj_matrix
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        参数:
            x: 输入特征 [B, T, V, C]
        返回:
            out: 输出特征 [B, T, V, C']
        """
        B, T, V, C = x.shape

        # 应用线性变换
        x = self.linear(x)  # [B, T, V, C']

        # 确保邻接矩阵和输入数据在同一设备上
        if self.adj_matrix.device != x.device:
            self.adj_matrix = self.adj_matrix.to(x.device)

        # 图卷积操作
        x = torch.einsum('vw,btwc->btvc', self.adj_matrix, x)

        return x


class PoseGCNTransformer(nn.Module):
    """基于GCN和Transformer的姿态分类模型"""
    def __init__(self, num_joints=133, in_channels=3, hidden_dim=128, num_classes=2,
                 num_gcn_layers=3, num_transformer_layers=6, feat_dim=64,num_heads=8):
        super().__init__()
        adj = get_adjacency_matrix(num_joints)
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        gcn_layers = []
        in_dim = in_channels
        for _ in range(num_gcn_layers):
            gcn_layers.append(SimpleGCNLayer(in_dim, hidden_dim, adj))
            gcn_layers.append(nn.ReLU())
            gcn_layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        self.gcn = nn.Sequential(*gcn_layers)

        # Transformer处理时间维度，输入需为 [B, T, hidden_dim]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.num_heads,
            #    batch_first=True,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ),
            num_layers=num_transformer_layers
        )

        # 特征提取层
        self.feat_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feat_dim)  # 输出用于对比学习的特征向量
        )

        # 分类器：输入维度为 feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # 输入维度: [B, T, V, C] = [4, 300, 133, 3]
        B, T, V, C = x.shape

        # GCN提取空间特征，输出: [B, T, V, hidden_dim] = [4, 300, 133, 128]
        x = self.gcn(x)

        # 空间维度池化（对V取平均）：[B, T, V, hidden_dim] → [B, T, hidden_dim]
        x = x.mean(dim=2)

        # Transformer提取时间特征：输入 [B, T, hidden_dim]
        x = self.transformer(x)  # 输出 [B, T, hidden_dim]

        # 时间维度池化（对T取平均）：[B, T, hidden_dim] → [B, hidden_dim]
        x = x.mean(dim=1)  # [B, hidden_dim]

        # 提取对比学习特征
        contrastive_features = self.feat_extractor(x)

        # 分类预测
        logits = self.classifier(contrastive_features)

        return logits, contrastive_features  # 返回分类结果和对比特征
