import os
import numpy as np
import torch
from torch_geometric.data import Data
from model import GCN
from dataset import get_dataset
import argparse
from torch_geometric.data import DataLoader
import abc_py
import time

def get_graph_data(state, lbl):
    base_path = '../project/'
    circuitName, actions = state.split('_')
    circuitPath = os.path.join(base_path, 'InitialAIG/train/' + circuitName + '.aig')   # NOTE only train dir ??应该是, test应该是用来设计aig的

    if not os.path.exists(os.path.join(base_path, 'train_aig_files', state + '.aig')):
        libFile = os.path.join(base_path, 'lib/7nm/7nm.lib')

        if not os.path.exists(circuitPath) or not os.path.exists(libFile):
            raise ValueError('path error')
        logFile = 'tmp.log'
        nextState = state + '.aig'  # current AIG file

        # Mapping action indices to their corresponding synthesis operations
        synthesisOpToPosDic = {
            0: "refactor",
            1: "refactor -z",
            2: "rewrite",
            3: "rewrite -z",
            4: "resub",
            5: "resub -z",
            6: "balance"
        }

        # Building the command string for synthesis operations
        actionCmd = ''
        for action in actions:
            actionCmd += (synthesisOpToPosDic[int(action)] + '; ')

        initial_abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + actionCmd + "read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile

        os.system(initial_abcRunCmd) 
        if not os.path.exists(os.path.join(base_path, 'train_aig_files/')):
            os.makedirs(os.path.join(base_path, 'train_aig_files/'))
        os.system(f"mv {state}.aig {os.path.join(base_path, 'train_aig_files/')}")
        # raise ValueError
    else:
        pass

    _abc = abc_py.AbcInterface()
    # abcpy.
    _abc.start()
    aig_path = os.path.join(os.path.join(base_path, 'train_aig_files', state + '.aig'))
    # _abc.read(state + '.aig')
    _abc.read(aig_path)
    data = {}

    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    node_features = np.zeros((numNodes, 2), dtype=float)
    
    
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
            node_features[nodeIdx, 0] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
            node_features[nodeIdx, 0] = 1
        else:
            data['node_type'][nodeIdx] = 2
            node_features[nodeIdx, 0] = 2
        
        if nodeType == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
            node_features[nodeIdx, 1] = 1
        elif nodeType == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2
            node_features[nodeIdx, 1] = 2
        else:
            node_features[nodeIdx, 1] = 0   # NOTE
            
        if aigNode.hasFanin0():     # 这个应该是接口吧
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
            
        if aigNode.hasFanin1():
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin) # "fanin"是指连接到一个逻辑门或电路节点的输入信号线的数量。换句话说，它表示一个逻辑门或电路节点接收的输入数量。对于一个逻辑门来说，它的输入线就是fanin。在布尔逻辑中，逻辑门的输入通常被称为"输入端"，而fanin表示逻辑门接收的输入端的数量。
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'], dtype=torch.float) # NOTE float
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['node_features'] = torch.tensor(node_features, dtype=torch.float)
    """
    `num_inverted_predecessors`是一个表示节点（AIG图中的节点）的反转输入的数量的特征。在AIG图中，每个节点可以有零个、一个或两个输入。如果一个节点的输入是反转的（即是一个NOT门），那么它的`num_inverted_predecessors`特征将表示输入中有多少个是反转的。

    在这个代码中，当节点类型为4时，`num_inverted_predecessors`被设置为1，表示该节点的一个输入是反转的；当节点类型为5时，`num_inverted_predecessors`被设置为2，表示该节点的两个输入都是反转的。
    """
    data['nodes'] = numNodes

    # print(data)

    node_features = data['node_features']   # Num node, 2
    edge_index = data['edge_index']

    graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([lbl]))        # to [1,]
    return graph_data

from tqdm import tqdm 
def train(model, device, dataset, optimizer, criterion):
    model.train()
    total_loss = 0
    # for data in dataset:
    for data in tqdm(dataset, desc='Training dataset'):
        state, lbl = data
        # lbl = lbl.to(device)
        optimizer.zero_grad()
        # print(lbl)
        
        graph_data = get_graph_data(state, lbl).to(device)
        output = model(graph_data)
        # print(graph_data.y.shape, ' ', output.shape)
        # print(graph_data.y)
        loss = criterion(output, graph_data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(output, ' ', graph_data.y)
    return total_loss / len(dataset)

def test(model, device, dataset, criterion):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for data in dataset:
            state, lbl = data
            graph_data = get_graph_data(state, lbl).to(device)
            output = model(graph_data)
            mse = criterion(output, graph_data.y).item()
            mae = torch.abs(output - graph_data.y).mean().item()
            total_mse += mse
            total_mae += mae
    avg_mse = total_mse / len(dataset)
    avg_mae = total_mae / len(dataset)
    return avg_mse, avg_mae

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Load dataset...')
    dataset = get_dataset(args.data)
    
    print('Load over !')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    model = GCN(num_node_features=2).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss() 

    for epoch in range(args.max_epoch):  
        time_start = time.time()
        train_loss = train(model, device, train_dataset, optimizer, criterion)
        mse, mae = test(model, device, test_dataset, criterion=criterion)
        print(f'Time: {time.time() - time_start:.2f} Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae: .4f}')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../project/project_data', help='The path of data in task 1')
    parser.add_argument('--num-node-features', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = args_parser()
    main(args)
