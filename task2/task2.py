# Importing required libraries
import os
import re
import numpy as np
import torch
import abc_py
from torch_geometric.data import Data
from model import GCN

synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

BASEPATH = '../project/'
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
LOGFILE = 'tmp.log'
LIBFILE = os.path.join(BASEPATH, 'lib/7nm/7nm.lib')

def cal_baseline(AIG, train=True, circuitPath=None, libFile=None):
    """根据 InitialAIG 里面的文件来获取 AIG 的 baseline"""
    state = AIG.split('.')[0]
    logFile = state + LOGFILE
    nextState = AIG  # current AIG file, 如 alu4.aig

    if circuitPath is None or libFile is None:
        if '_' in state: circuitName, actions = state.split('_')
        else: circuitName = state
        middle_dir = 'train' if train else 'test'
        circuitPath = os.path.join(BASEPATH, f'InitialAIG/{middle_dir}/' + circuitName + '.aig')
        libFile = LIBFILE

    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + "read_lib " + libFile + "; write " + nextState + "; write_bench -l " + nextState + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    print("baseline:", baseline)
    return baseline

def evaluate_AIG(AIG, train=True):
    """根据 InitialAIG 里面的文件来获取 AIG 的 regularized score"""
    state = AIG.split('.')[0]
    if '_' in state: circuitName, actions = state.split('_')
    else: circuitName = state
    middle_dir = 'train' if train else 'test'
    circuitPath = os.path.join(BASEPATH, f'InitialAIG/{middle_dir}/' + circuitName + '.aig')
    libFile = LIBFILE
    logFile = state + LOGFILE
    # yosys-abc -c "read ../project/InitialAIG/train/alu2.aig; read_lib ../project/lib/7nm/7nm.lib; map; topo; stime" > alu2.log
    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        eval = float(areaInformation[-9]) * float(areaInformation[-4])
    baseline = cal_baseline(AIG, train, circuitPath, libFile)
    regularized_eval = 1 - eval / baseline
    print("eval:",regularized_eval)
    return regularized_eval

def get_pkl_data():
    """
    从 project_data2 的 .pkl 文件里读取有关 AIG 的数据
    AIG: such as alu2_0362351640.aig
    每个这样的 AIG 文件对应一个 .pkl 文件，如 adder_1010.pkl 对应 adder_4000021242.aig
    每个 .pkl 文件里有一个字典，包含了 adder_, adder_4, ..., adder_4000021242 这 10 个 step 和其对应的所有 rewards (len=10)
    rewards 就是该步骤的 score 减去第 10 个 step 的 score 的差值，所以说 rewards[9] = 0，必定为 0
    感觉搜索的目的，就是尽可能地在最小的 step i 下，让 rewards[i] = 0
    """
    import pickle
    import os
    data_ls = os.listdir('../project/project_data2')

    AIG_data = {}
    for file in data_ls:
        with open(os.path.join('../project/project_data2', file), 'rb') as f:
            data = pickle.load(f)
            AIG_data[data['target'][9]] = data['target']
    return AIG_data

# alu2.aig 在 train 文件夹里面, train=True
# alu4.aig 在 test 文件夹里面, test=False
#evaluate_AIG('alu2.aig', train=True)

def predict_reward(AIG):
    """AIG: such as alu2_32.aig"""
    return np.random.rand()

def main():
    AIG = 'alu4.aig'
    libFile = LIBFILE
    logFile = 'tmp.log'

    for step in range(10):
        childs = []
        cur_state = AIG.split('.')[0]
        cur_state = cur_state + '_' if '_' not in cur_state else cur_state  # in the beginning

        childScores = []
        for child in range(7):
            childFile = cur_state + str(child) + '.aig'
            predicted = predict_reward(childFile)       # 要用模型来预测 reward
            childScores.append(predicted)
            childs.append(childFile)
        action = np.argmin(childScores)
        AIG = childs[action]

    print(AIG)

    # FIXME 此处的 AIG 如 alu4_5250010256.aig，InitialAIG 不存在这种文件，要如何生成？
    abcRunCmd = "yosys-abc -c \"read " + AIG + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    print(abcRunCmd)
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        adpVal = float(areaInformation[-9]) * float(areaInformation[-4])

main()

