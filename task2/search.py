import numpy as np
from collections import deque
from sortedcontainers import SortedDict
from tqdm import tqdm
from model import DeeperEnhancedGCN
import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import abc_py

class Search(object):
    def __init__(self, n_steps=10, n_branch=7):
        self.set_param(n_steps, n_branch)

    def predict_fn(self, AIG):
        model_predict_now = DeeperEnhancedGCN(num_node_features=2).cuda()
        model_predict_future = DeeperEnhancedGCN(num_node_features=2).cuda()
        model_predict_now.load_state_dict(torch.load('./model_final/now.pth'))
        model_predict_future.load_state_dict(torch.load('./model_final/future.pth'))
        
        base_path = '../project'
        # print(f'\n here {AIG}')
        if not '_' in AIG:
            AIG = AIG.split('.')[0] + '_'
        if '.aig' in AIG:
            AIG = AIG.replace('.aig', '')
            
        circuitName, actions = AIG.split('_')
        circuitPath = os.path.join(base_path, 'InitialAIG/test/' + circuitName + '.aig')   # NOTE only train dir ??应该是, test应该是用来设计aig的

        if not os.path.exists(os.path.join(base_path, 'test_aig_files', AIG + '.aig')):
            libFile = os.path.join(base_path, 'lib/7nm/7nm.lib')

            if not os.path.exists(circuitPath) or not os.path.exists(libFile):
                raise ValueError('path error')
            logFile = 'tmp.log'
            nextState = AIG + '.aig'  # current AIG file

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
            # print('\n here', AIG)
            for action in actions:
                actionCmd += (synthesisOpToPosDic[int(action)] + '; ')

            initial_abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + actionCmd + "read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile

            # NOTE use the same aig file !!!! in task 1 and task 2 (xxx.aig 对应的aig相同)
            os.system(initial_abcRunCmd) 
            if not os.path.exists(os.path.join(base_path, 'test_aig_files/')):
                os.makedirs(os.path.join(base_path, 'test_aig_files/'))
            os.system(f"mv {AIG}.aig {os.path.join(base_path, 'test_aig_files/')}")
            # raise ValueError
        else:
            pass
        
        
        _abc = abc_py.AbcInterface()
        # abcpy.
        _abc.start()
        aig_path = os.path.join(os.path.join(base_path, 'test_aig_files', AIG + '.aig'))
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

        graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([-1.0]))        # to [1,]
        
        dataloader = DataLoader([graph_data], batch_size=1)
        
        for itm in dataloader:
            graph_data = itm.cuda()
            cur_aig_score = model_predict_now(graph_data).item()
            future_aig_score = model_predict_future(graph_data).item()
        # print(cur_aig_score, future_aig_score, ' score')
        # raise ValueError
        return cur_aig_score + future_aig_score
        
    
    def set_param(self, n_steps=None, n_branch=None):
        # self.predict_fn = predict_fn if predict_fn is not None else self.predict_fn
        self.n_steps = n_steps if n_steps is not None else self.n_steps
        self.n_branch = n_branch if n_branch is not None else self.n_branch

    def __call__(self, AIG, method='greedy'):
        if method == 'greedy':
            return self.greedy(AIG)
        elif method == 'DFS':
            return self.DFS(AIG)
        elif method == 'BFS':
            return self.BFS(AIG)
        elif method == 'BestFirstSearch':
            return self.BestFirstSearch(AIG)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def greedy(self, AIG):
        pbar = tqdm(desc='Greedy search', leave=True, ascii=True, unit=' step')

        AIG = AIG.split('.')[0] + '_.' + AIG.split('.')[1]
        for step in range(self.n_steps):
            childs = []
            cur_state = AIG.split('.')[0]

            childScores = []
            for child in range(self.n_branch):
                childFile = cur_state + str(child) + '.aig'
                predicted = self.predict_fn(childFile)
                childScores.append(predicted)
                childs.append(childFile)
                pbar.update(1)
            action = np.argmax(childScores)
            AIG = childs[action]
        return AIG
    
    def DFS(self, AIG):
        pbar = tqdm(desc='DFS search', leave=True, ascii=True, unit=' step')

        AIG = AIG.split('.')[0] + '_.' + AIG.split('.')[1]
        dq = deque()
        dq.append((AIG, self.predict_fn(AIG)))
        max_value = float('-inf')
        max_AIG = None
        
        try:
            while dq:
                cur, predicted = dq.pop()
                cur_state = cur.split('.')[0]
                cur_len = len(cur_state.split('_')[-1])

                if cur_len == self.n_steps: 
                    if predicted > max_value:
                        max_value = predicted
                        max_AIG = cur
                    continue

                childs = []
                for child in range(self.n_branch):
                    childFile = cur_state + str(child) + '.aig'
                    childs.append(childFile)
                    pbar.update(1)

                predicted = []
                for child in childs:
                    predicted.append(self.predict_fn(child))
                dq.extend(zip(childs, predicted))
                pbar.set_description_str(f'DFS deque {len(dq)}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG
    
    def BFS(self, AIG):
        pbar = tqdm(desc='BFS search', leave=True, ascii=True, unit=' step')
        
        AIG = AIG.split('.')[0] + '_.' + AIG.split('.')[1]
        dq = deque()
        dq.append((AIG, self.predict_fn(AIG)))
        max_value = float('-inf')
        max_AIG = None

        try:
            while dq:
                cur, predicted = dq.popleft()
                cur_state = cur.split('.')[0]
                cur_len = len(cur_state.split('_')[-1])

                if cur_len == self.n_steps: 
                    if predicted > max_value:
                        max_value = predicted
                        max_AIG = cur
                    continue

                childs = []
                for child in range(self.n_branch):
                    childFile = cur_state + str(child) + '.aig'
                    childs.append(childFile)
                    pbar.update(1)

                predicted = []
                for child in childs:
                    predicted.append(self.predict_fn(child))
                dq.extend(zip(childs, predicted))
                pbar.set_description_str(f'BFS deque {len(dq)}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG
    
    def BestFirstSearch(self, AIG, maxsize=2000):
        pbar = tqdm(desc='BestFS search', leave=True, ascii=True, unit=' step')

        AIG = AIG.split('.')[0] + '_.' + AIG.split('.')[1]
        sd = SortedDict()
        sd[self.predict_fn(AIG)] = AIG
        max_value = float('-inf')
        max_AIG = None

        try:
            while len(sd) > 0:
                predicted, cur = sd.popitem(index=-1)
                cur_state = cur.split('.')[0]
                cur_len = len(cur_state.split('_')[-1])

                if cur_len == self.n_steps: 
                    if predicted > max_value:
                        max_value = predicted
                        max_AIG = cur
                    continue

                childs = []
                for child in range(self.n_branch):
                    childFile = cur_state + str(child) + '.aig'
                    childs.append(childFile)
                    pbar.update(1)
                
                predicted = []
                for child in childs:
                    predicted.append(self.predict_fn(child))
                
                for i in range(self.n_branch):
                    if len(sd) == maxsize: sd.popitem(index=0)
                    sd[predicted[i]] = childs[i]
                pbar.set_description_str(f'BestFirstSearch pq {len(sd)} --- {max_AIG} --- {max_value}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG