import pickle 
import os
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
import abc_py
import numpy as np
import random

# class dataset(Data.Dataset):
#     def __init__(self, path):
#         data_ls = os.listdir(path)
#         self.aig_names = []
#         self.labels = []
        
#         cnt = 0
#         for data_file in tqdm(data_ls, desc="Loading data"):  # Use tqdm for progress bar
#             cnt += 1
#             # if cnt >= 100:
#             #     break
#             data_path = os.path.join(path, data_file)
#             with open(data_path, 'rb') as file:
#                 data = pickle.load(file)
#             aig_name = data['input']
#             label = data['target']
#             self.aig_names.extend(aig_name)
#             self.labels.extend(label)
            
#         # raise NotImplementedError
    
#     def __getitem__(self, index):
#         return self.aig_names[index], self.labels[index]
    
#     def __len__(self):
#         return len(self.labels)

class PYGDataset(InMemoryDataset):
    def __init__(self, path, transform=None, pre_transform=None):
        self.root = os.path.join(path, '..', 'pyg')
        self.path = path
        super(PYGDataset, self).__init__(self.root, transform, pre_transform)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.path)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        data_ls = os.listdir(self.path)
        data_ls = random.sample(data_ls, 20)
        self.data_ls = data_ls

        for data_file in tqdm(data_ls, desc="Processing data"):
            data_path = os.path.join(self.path, data_file)
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            aig_names = data['input']
            labels = data['target']

            for aig_name, label in zip(aig_names, labels):
                graph_data = self.get_graph_data(aig_name, label)
                data_list.append(graph_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def get_graph_data(self, state, lbl):
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

        for nodeIdx in range(numNodes):
            aigNode = _abc.aigNode(nodeIdx)
            nodeType = aigNode.nodeType()
            data['num_inverted_predecessors'][nodeIdx] = 0
            if nodeType == 0 or nodeType == 2:
                data['node_type'][nodeIdx] = 0
            elif nodeType == 1:
                data['node_type'][nodeIdx] = 1
            else:
                data['node_type'][nodeIdx] = 2
            
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
                
            if aigNode.hasFanin0():
                fanin = aigNode.fanin0()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
            if aigNode.hasFanin1():
                fanin = aigNode.fanin1()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin) # "fanin"是指连接到一个逻辑门或电路节点的输入信号线的数量。换句话说，它表示一个逻辑门或电路节点接收的输入数量。对于一个逻辑门来说，它的输入线就是fanin。在布尔逻辑中，逻辑门的输入通常被称为"输入端"，而fanin表示逻辑门接收的输入端的数量。
        data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
        data['node_type'] = torch.tensor(data['node_type'], dtype=torch.float).reshape(-1, 1) # NOTE float
        data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
        """
        `num_inverted_predecessors`是一个表示节点（AIG图中的节点）的反转输入的数量的特征。在AIG图中，每个节点可以有零个、一个或两个输入。如果一个节点的输入是反转的（即是一个NOT门），那么它的`num_inverted_predecessors`特征将表示输入中有多少个是反转的。

        在这个代码中，当节点类型为4时，`num_inverted_predecessors`被设置为1，表示该节点的一个输入是反转的；当节点类型为5时，`num_inverted_predecessors`被设置为2，表示该节点的两个输入都是反转的。
        """
        data['nodes'] = numNodes

        # print(data)

        node_features = data['node_type']   # FIXME 这里需要revise
        edge_index = data['edge_index']

        graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(lbl))
        return graph_data
    
    def __len__(self):
        return len(self.data.y)

def get_dataset(path):
    return PYGDataset(path)

if __name__ == '__main__':
    get_dataset('../project/project_data')