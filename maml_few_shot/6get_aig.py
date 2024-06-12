import os
import abc_py
import random

from tqdm import tqdm

import numpy as np

random.seed(42)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import dill
import pickle
import os
import socket 

import time

import warnings

warnings.filterwarnings("ignore")

pid = os.getpid()

# 输出 PID
print("PID:", pid)

import argparse

# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--rank', type=int, default=-1, required=False)

# 解析命令行参数
args = parser.parse_args()
print(args)


# 设置文件夹的路径
folder_path = '/Disk2/xiaoyang/ML/project/project_data'

# 获取文件夹内所有文件的路径
files_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

# 打印文件列表
# print(files_list)
print(len(files_list))

file_type = [item.split('/')[-1].split('_')[0] for item in files_list]

# 去除重复的文件类型
file_type = list(set(file_type))
file_type.sort()
print(file_type)
print(len(file_type))

# 划分训练集和验证集
train_files = []
val_files = []

# 随机选取一部分文件type作为训练集
train_type = random.sample(file_type, 23)
print(train_type)

rank = args.rank
train_type = [train_type[rank]]

# 其余为val set
val_type = list(set(file_type) - set(train_type))
print(val_type)

# 根据文件名划分为训练集和验证集
train_files = [file for file in files_list if file.split('/')[-1].split('_')[0] in train_type]
val_files = [file for file in files_list if file.split('/')[-1].split('_')[0] in val_type]

bb = len(train_files)
print(len(train_files))
print(train_type)
print(len(val_files))

X_train = []
Y_train = []

synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

import pickle as pkl

train_files = random.sample(train_files, bb// 7)
print("Actual train files: ", len(train_files))

# 手动实现一个时钟，每到特定的iter就print ETA

tot = len(train_files)
cnt = 0

start_time = time.time()

for item in train_files:
    cnt += 1

    if (cnt % 5 == 0):
        end_tme = time.time()
        print(f'ETA: {(end_tme - start_time) / cnt * (tot - cnt)}')
        print(f'Progress: {cnt}/{tot}')

    with open(item, 'rb') as f:
        data_pth = pkl.load(f)
    
    # print(data_pth)    

    for src, tgt in zip(data_pth['input'], data_pth['target']):
        # print(src, tgt)
        # assert 0

        circuitName, actions = src.split('_')
        circuitPath = '/Disk2/xiaoyang/ML/project/InitialAIG/train/' + circuitName + '.aig'
        libFile = '/Disk2/xiaoyang/ML/project/lib/7nm/7nm.lib'
        # logFile = '/Disk2/xiaoyang/ML/project/alu2.log'
        nextState = f'task_t/tmp{rank}.aig' # current AIG file

        action_cmd = ''
        for action in actions:
            action_cmd += (synthesisOpToPosDic[int(action)] + '; ')

        abcRunCmd = "../yosys/yosys-abc -c \"read " + circuitPath + "; " + action_cmd + "read_lib " + libFile + "; write " + nextState + f"; print_stats\" > task_t/out{rank}.log"
        # print(abcRunCmd)
        os.system(abcRunCmd)

        # assert 0

        _abc = abc_py.AbcInterface()
        _abc.start()
        _abc.read(nextState)
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
            if (aigNode.hasFanin0()):
                fanin = aigNode.fanin0()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
            if (aigNode.hasFanin1()):
                fanin = aigNode.fanin1()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)

        data['edge_index'] = np.array([edge_src_index, edge_target_index], dtype=int)
        data['node_type'] = np.array(data['node_type'])
        data['num_inverted_predecessors'] = np.array(data['num_inverted_predecessors'])
        data['nodes'] = numNodes

        X_train.append(data)
        Y_train.append(tgt)

with open(f'X_train_{rank}_{train_type[0]}.pkl', 'wb') as f:
    pkl.dump(X_train, f)

with open(f'Y_train_{rank}_{train_type[0]}.pkl', 'wb') as f:
    pkl.dump(Y_train, f)

    # print(X_train)
    # print(Y_train)
    # assert 0

        # ../yosys/yosys-abc -c "read /Disk2/xiaoyang/ML/project/InitialAIG/train/adder.aig; resub; rewrite; read_lib /Disk2/xiaoyang/ML/project/lib/7nm/7nm.lib; write adder_42.aig; print_stats" > /Disk2/xiaoyang/ML/project/alu2.log

        # abcRunCmd = "../yosys/yosys"
        
        # X_train.append(input)
        # Y_train.append(data['output'])




