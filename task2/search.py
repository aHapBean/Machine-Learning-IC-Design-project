import numpy as np
from collections import deque
from sortedcontainers import SortedDict
from tqdm import tqdm
from model import DeeperEnhancedGCN
import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from predict import Predict
import abc_py 

class Search(object):
    def __init__(self, n_steps=10, n_branch=7, predict_fn=None):
        self.set_param(n_steps, n_branch, predict_fn)
    
    def set_param(self, n_steps=10, n_branch=7, predict_fn=None):
        # self.predict_fn = predict_fn if predict_fn is not None else self.predict_fn
        assert predict_fn is not None, 'predict_fn is None ERROR!!'
        self.n_steps = n_steps
        self.n_branch = n_branch
        self.predict_fn = predict_fn


    def __call__(self, AIG, method='greedy', maxsize=200):
        if method == 'greedy':
            output_AIG = self.greedy(AIG)
        elif method == 'DFS':
            output_AIG = self.DFS(AIG)
        elif method == 'BFS':
            output_AIG = self.BFS(AIG)
        elif method == 'BestFirstSearch':
            output_AIG = self.BestFirstSearch(AIG, maxsize)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
        return output_AIG
        # actions = output_AIG.split('_')[1].split('.')[0]
        # base = output_AIG.split('_')[0] + '_'
        # max_score = float('-inf')
        # max_AIG = None
        # for action in actions:
        #     base += action
        #     score = self.predict_fn(base + '.aig')
        #     if score > max_score:
        #         max_score = score
        #         max_AIG = base + '.aig'
        # return max_AIG

    def greedy(self, AIG):
        pbar = tqdm(desc='Greedy search', leave=True, ascii=True, unit=' step')
        
        if not '_' in AIG:
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
        if not '_' in AIG:
            AIG = AIG.split('.')[0] + '_.' + AIG.split('.')[1]
        
        dq = deque()
        dq.append((AIG, self.predict_fn(AIG)))
        max_value = float('-inf')
        max_AIG = None
        
        try:
            while dq:
                cur, predicted = dq.pop()   # 尾部删除
                cur_state = cur.split('.')[0]
                cur_len = len(cur_state.split('_')[-1])

                if predicted > max_value:
                    max_value = predicted
                    max_AIG = cur
                if cur_len == self.n_steps: 
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
        
        if not '_' in AIG:
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

                if predicted > max_value:
                    max_value = predicted
                    max_AIG = cur
                if cur_len == self.n_steps: 
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

        if not '_' in AIG:
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

                if predicted > max_value:
                    max_value = predicted
                    max_AIG = cur
                if cur_len == self.n_steps: 
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