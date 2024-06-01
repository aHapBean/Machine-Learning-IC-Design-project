import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""


"""

# class GCN(torch.nn.Module):
#     def __init__(self, num_node_features):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 16)
#         self.conv2 = GCNConv(16, 1)  # 输出层现在只有一个节点

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         """
#         data.x : bs, num_nodes, num_node_features
#         edge_index : 2, num_edges
#         """
#         # print(x.shape)
#         x = x.unsqueeze(-1).float()
#         print(x)    #  device='cuda:0', dtype=torch.float64)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x  # 不使用激活函数，直接返回结果，适合回归任务

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 1)  # 输出一个值

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print()
        # x = x.unsqueeze(-1).float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))  # 模拟单个图的情况
        x = self.fc(x)
        return x
