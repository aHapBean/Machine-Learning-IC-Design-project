import pickle as pkl
from tqdm import tqdm

import time
from line_profiler import LineProfiler
import copy

import random
random.seed(42)
import higher

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

torch.manual_seed(42)

X_file_names = [
    "X_train_0_priority.pkl",
    "X_train_1_apex5.pkl",
    "X_train_2_adder.pkl",
    "X_train_3_c5315.pkl",
    "X_train_4_c2670.pkl",
    "X_train_5_max512.pkl",
    "X_train_6_arbiter.pkl",
    "X_train_7_prom2.pkl",
    "X_train_8_ctrl.pkl",
    "X_train_9_frg1.pkl",
    # "X_train_10_multiplier.pkl",
    "X_train_11_alu2.pkl",
    "X_train_12_c6288.pkl",
    "X_train_13_c1355.pkl",
    "X_train_14_table5.pkl",
    "X_train_15_i7.pkl",
    "X_train_16_max.pkl",
    "X_train_17_i8.pkl",
    "X_train_18_b2.pkl",
    "X_train_19_int2float.pkl",
    "X_train_20_apex3.pkl",
    # "X_train_21_log2.pkl",
    "X_train_22_m3.pkl"
]

Y_file_names = [
    "Y_train_0_priority.pkl",
    "Y_train_1_apex5.pkl",
    "Y_train_2_adder.pkl",
    "Y_train_3_c5315.pkl",
    "Y_train_4_c2670.pkl",
    "Y_train_5_max512.pkl",
    "Y_train_6_arbiter.pkl",
    "Y_train_7_prom2.pkl",
    "Y_train_8_ctrl.pkl",
    "Y_train_9_frg1.pkl",
    # "Y_train_10_multiplier.pkl",
    "Y_train_11_alu2.pkl",
    "Y_train_12_c6288.pkl",
    "Y_train_13_c1355.pkl",
    "Y_train_14_table5.pkl",
    "Y_train_15_i7.pkl",
    "Y_train_16_max.pkl",
    "Y_train_17_i8.pkl",
    "Y_train_18_b2.pkl",
    "Y_train_19_int2float.pkl",
    "Y_train_20_apex3.pkl",
    # "Y_train_21_log2.pkl",
    "Y_train_22_m3.pkl"
]

bs_adapt = 2
Adapt_sample = []

X_feature = []
Y_feature = []    

for item in tqdm(X_file_names[:18],):
    with open(item, "rb") as f:
        X_feature.append(pkl.load(f))

for item in tqdm(Y_file_names[:18]):
    with open(item, "rb") as f:
        Y_feature.append(pkl.load(f))
       
print('Train files', len(X_feature))
assert len(Y_feature) == len(X_feature)

train_datas = []

for X, Y in zip(X_feature, Y_feature):
    assert len(X) == len(Y)
    datas = []

    for data, label in zip(X, Y):
        edge_index = torch.tensor(data['edge_index'], device='cuda').to(torch.long)
        x = torch.stack([torch.tensor(data['node_type'], device='cuda'),torch.tensor(data['num_inverted_predecessors'],device='cuda')], dim=-1).to(torch.float32)
        y = torch.tensor(label, device='cuda')

        assert x.shape[0] == data['nodes']
        num_nodes = x.shape[0]
        
        datas.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=label))

    train_datas.append(datas)

print('Load train data done:', len(train_datas))

X_feature = []
Y_feature = []

Ad_feature = []
Ad_label = []

for item in tqdm(X_file_names[-3:],):
    with open(item, "rb") as f:
        tt = pkl.load(f)
        Ad_feature.extend(tt[:bs_adapt])
        X_feature.extend(tt)

for item in tqdm(Y_file_names[-3:]):
    with open(item, "rb") as f:
        tt = pkl.load(f)
        Ad_label.extend(tt[:bs_adapt])
        Y_feature.extend(tt)


print('Val files', len(X_feature))
assert len(Y_feature) == len(X_feature)

val_data = []

for data, label in zip(X_feature, Y_feature):
    edge_index = torch.tensor(data['edge_index'], device='cuda').to(torch.long)
    x = torch.stack([torch.tensor(data['node_type'], device='cuda'),torch.tensor(data['num_inverted_predecessors'],device='cuda')], dim=-1).to(torch.float32)
    y = torch.tensor(label, device='cuda')

    assert x.shape[0] == data['nodes']
    num_nodes = x.shape[0]
    
    val_data.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=label))

for data, label in zip(Ad_feature, Ad_label):
    edge_index = torch.tensor(data['edge_index'], device='cuda').to(torch.long)
    x = torch.stack([torch.tensor(data['node_type'], device='cuda'),torch.tensor(data['num_inverted_predecessors'],device='cuda')], dim=-1).to(torch.float32)
    y = torch.tensor(label, device='cuda')

    assert x.shape[0] == data['nodes']
    num_nodes = x.shape[0]
    
    Adapt_sample.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=label))

print('Load val data done:', len(val_data))
print('Load adapt data done:', len(Adapt_sample))

print("=========================")
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        # self.dropout = torch.nn.Dropout(p=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        for (name, param) in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            if 'fc' in name and 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # 聚合所有节点特征成图级特征
        
        x = self.fc(x)
        # x = 2 * torch.sigmoid(x) - 1  # 输出值在 -1 和 1 之间
        x = x.squeeze(-1)
        return x

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout_rate=0.0):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels // 8, heads=1, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_channels // 8, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        x = 2 * torch.sigmoid(x) - 1  # 输出值在 -1 和 1 之间
        x = x.squeeze(-1)
        return x

model = GCN(input_dim=2, hidden_dim=64, output_dim=1).to('cuda')
# model = GATModel(in_channels=2, hidden_channels=64, out_channels=1, heads=4, dropout_rate=0.1).to('cuda')

bs_support_cate = 4
bs_support_train = 10
bs_support_per = 32
bs_task = 8
bs_val = 32
val_step = 10
update_steps = 1

val_loader = DataLoader(val_data, batch_size=bs_val, shuffle=False)
print('Val loader:', len(val_loader)) # 除以bs之后的长度

optimizer_outer = torch.optim.Adam(model.parameters(), lr=0.01,)
optimizer_inner = torch.optim.Adam(model.parameters(), lr=0.01,)
criterion_1 = torch.nn.L1Loss()
criterion_2 = torch.nn.MSELoss()
criterion = torch.nn.SmoothL1Loss(beta=0.05)

def get_data(data_list, iter):
    dd_train = []
    dd_test = []

    for i in range(bs_support_per):
        if iter + i >= len(data_list):
            return -1, -1

        if i < bs_support_train:
            dd_train.append(data_list[iter + i])
        else:
            dd_test.append(data_list[iter + i])

    return dd_train, dd_test


def inner_updata(model, support_set, lr=1e-2):
    outputs = model(support_set)

    target = support_set.y
    loss = criterion(outputs, target)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    updated_params = {name: param - lr * grad for ((name, param), grad) in zip(model.named_parameters(), grads)}

    return updated_params

def compute_loss_with_updated_params(model, query_set, updated_params):
    for name, param in model.named_parameters():
        param.data = updated_params[name]

    outputs = model(query_set)
    target = query_set.y
    loss = criterion(outputs, target)

    return loss

def train_one_epoch(epoch):

    for data_ in train_datas:
        random.shuffle(data_)

    data_pool = train_datas

    model.train()

    start_time = time.time()

    iter_pool = [0 for i in range(18)]

    cnt = 0

    flag = False

    total_loss = 0

    while True:
        optimizer_outer.zero_grad()

        cnt += 1
        if cnt % 10 == 0:
            print(f'Epoch: {epoch:03d}, Iter: {cnt: 04d}, Train Loss: {total_loss / cnt:.4f}, Time: {time.time() - start_time:.4f}')

        task_pool = []

        for i in range(bs_task):
            tmp_selected = random.sample(range(0, 18), bs_support_cate)
            # print(tmp_selected)
            data_pool_train = []
            data_pool_test = []

            for idx in tmp_selected:
                ans1, ans2 = get_data(data_pool[idx], iter_pool[idx])
                if ans1 == -1:
                    print(f'Epoch: {epoch:03d}, Iter: {cnt: 04d}, Train Loss: {total_loss / cnt:.4f}, Time: {time.time() - start_time:.4f}')
                    return total_loss / cnt

                iter_pool[idx] += bs_support_per
                data_pool_train.append(ans1)  
                data_pool_test.append(ans2)           

            task_pool.append((data_pool_train, data_pool_test))

        # local_loss = 0
        meta_loss = 0.0

        for task in task_pool:            
            all_data_train = []
            all_data_test = []

            data_pool_train, data_pool_test = task
            for data_train, data_test in zip(data_pool_train, data_pool_test):
                assert len(data_train) == bs_support_train
                assert len(data_test) == bs_support_per - bs_support_train

                all_data_train.extend(data_train)
                all_data_test.extend(data_test)
            # all_data_train.extend(data_train)
            # all_data_test.extend(data_test)

            data_train = Batch.from_data_list(all_data_train).to('cuda')
            data_test = Batch.from_data_list(all_data_test).to('cuda')

            with higher.innerloop_ctx(model, optimizer_inner,  copy_initial_weights=True) as (fmodel, diffopt):
                for i in range(update_steps):
                    outputs = fmodel(data_train)
                    target = data_train.y
                    loss = criterion(outputs, target)
                    diffopt.step(loss)

                outputs = fmodel(data_test)
                target = data_test.y
                query_loss = criterion(outputs, target)
                meta_loss += loss

            # model_inner.load_state_dict(model.state_dict())

            # updated_params = inner_updata(model_inner, data_train, lr=1e-2)

            # loss = compute_loss_with_updated_params(model_inner, data_test, updated_params)

        meta_loss /= bs_task
        total_loss += meta_loss.item()

        meta_loss.backward()
        optimizer_outer.step()

def evaluate(loader):
    # model.eval()

    adapt_data = Batch.from_data_list(Adapt_sample).to('cuda')

    # Adapt
    model_inner = copy.deepcopy(model)
    # model_inner.train()
    optimizer_test = torch.optim.Adam(model_inner.parameters(), lr=0.002,)

    for i in range(val_step):
        optimizer_test.zero_grad()
        tgt = model_inner(adapt_data)
        loss = criterion(tgt, adapt_data.y)
        # print(adapt_data.y)
        # print(f'Adapt Step: {i+1:03d}, Loss: {loss:.4f}')
        loss.backward()
        optimizer_test.step()
        
        center = 0
        var = 0

        # 类似计算L2,L1偏移量
        with torch.no_grad():
            for data in loader:
                data = data.to('cuda')
                pred = model_inner(data)
                center += criterion_1(pred, data.y)
                var += criterion_2(pred, data.y)

        print(f'Val Step: {i+1:03d}, Center: {center / len(loader):.4f}, Var: {var / len(loader):.4f}')
    # return center / len(loader), var / len(loader)

for epoch in range(100):
    loss = train_one_epoch(epoch)
    evaluate(val_loader)