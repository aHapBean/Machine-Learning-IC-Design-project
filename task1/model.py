import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 1)  # 输出一个值

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch) 
        x = self.fc(x).flatten()
        return x

# NOTE 可以用两种模型对比性能

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, JumpingKnowledge

# NOTE 加 batchnorm 不行
# from torch_geometric.nn.norm import BatchNorm

# class EnhancedGCN(torch.nn.Module):
#     def __init__(self, num_node_features):
#         super(EnhancedGCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 16)
#         self.bn1 = BatchNorm(16)  # 添加BatchNorm层
#         self.conv2 = GATConv(16, 16, heads=8)
#         self.bn2 = BatchNorm(16 * 8)  # 添加BatchNorm层
#         self.jump = JumpingKnowledge("cat")
#         self.fc1 = torch.nn.Linear(16 * 8 + 16, 64)  # 由于JumpingKnowledge的cat模式，维度变化
#         self.bn3 = BatchNorm(64)  # 添加BatchNorm层
#         self.fc2 = torch.nn.Linear(64, 1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)  # 应用BatchNorm
#         x1 = F.relu(x)
        
#         x = self.conv2(x1, edge_index)
#         x = self.bn2(x)  # 应用BatchNorm
#         x2 = F.relu(x)
        
#         x_jump = self.jump([x1, x2])
#         x_pool = global_mean_pool(x_jump, batch)
        
#         x = self.fc1(x_pool)
#         x = F.relu(x)
        
#         x = self.fc2(x).flatten()
#         return x


class EnhancedGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnhancedGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GATConv(16, 16, heads=8)  # 使用注意力机制
        self.jump = JumpingKnowledge("cat")
        self.fc1 = torch.nn.Linear(16 + 8 * 16, 16)  # 跳跃连接后的维度变化
        self.fc2 = torch.nn.Linear(16, 1)  # 输出一个值

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.shape, ' x')
        x1 = F.relu(self.conv1(x, edge_index))
        # print(x1.shape, ' x1')  # N, 16
        x2 = F.relu(self.conv2(x1, edge_index))
        # print(x2.shape, ' x2')  # N, 128 = 8 * 16
        x_jump = self.jump([x1, x2])  # 跳跃连接
        x_pool = global_mean_pool(x_jump, batch)
        # print(x_pool.shape)
        x = F.relu(self.fc1(x_pool))
        x = self.fc2(x).flatten()
        return x

"""
https://www.cnblogs.com/picassooo/p/15437658.html
不管bs多大，会合成一个大图?
    def forward(self, g):
        g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        # 为方便，我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float() # [N, 1]
        # 执行图卷积和激活函数
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h    # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')   # [n, hidden_dim]
        return self.classify(hg)  # [n, n_classes]
"""