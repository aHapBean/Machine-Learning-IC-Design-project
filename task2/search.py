import numpy as np
from collections import deque
from sortedcontainers import SortedDict
from tqdm import tqdm

class Search(object):
    def __init__(self, predict_fn, n_steps=10, n_branch=7):
        assert predict_fn is not None
        self.set_param(predict_fn, n_steps, n_branch)

    def set_param(self, predict_fn=None, n_steps=None, n_branch=None):
        self.predict_fn = predict_fn if predict_fn is not None else self.predict_fn
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
        pbar = tqdm(desc='Greedy search', leave=True, ascii=True, unit=' step')

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

                predicted = self.predict_fn(childs)
                dq.extend(zip(childs, predicted))
                pbar.set_description_str(f'DFS deque {len(dq)}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG
    
    def BFS(self, AIG):
        pbar = tqdm(desc='Greedy search', leave=True, ascii=True, unit=' step')
        
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

                predicted = self.predict_fn(childs)
                dq.extend(zip(childs, predicted))
                pbar.set_description_str(f'BFS deque {len(dq)}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG
    
    def BestFirstSearch(self, AIG, maxsize=2000):
        pbar = tqdm(desc='Greedy search', leave=True, ascii=True, unit=' step')

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
                predicted = self.predict_fn(childs)
                for i in range(self.n_branch):
                    if len(sd) == maxsize: sd.popitem(index=0)
                    sd[predicted[i]] = childs[i]
                pbar.set_description_str(f'BestFirstSearch pq {len(sd)} --- {max_AIG} --- {max_value}')
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        return max_AIG