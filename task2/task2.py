# Importing required libraries
import os
import re
import numpy as np
import torch
import abc_py
from torch_geometric.data import Data
from model import GCN, DeeperEnhancedGCN
from search import Search

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
    logFile = os.path.join('libFile', state + LOGFILE)
    nextState = os.path.join('aigFile', AIG)  

    if circuitPath is None or libFile is None:
        if '_' in state: circuitName, actions = state.split('_')
        else: circuitName = state
        middle_dir = 'train' if train else 'test'
        circuitPath = os.path.join(BASEPATH, f'InitialAIG/{middle_dir}/' + circuitName + '.aig')
        libFile = LIBFILE

    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + "read_lib " + libFile + "; write " + nextState + "; write_bench -l " + nextState + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    # yosys-abc -c "read ../project/InitialAIG/test/c880.aig; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;read_lib ../project/lib/7nm/7nm.lib; write aigFile/c880_5243.aig; write_bench -l aigFile/c88      "ubun" 21:11 06-6月-240_5243.aig; map; topo; stime" > libFile/c880_5243tmp.log
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    #print("baseline:", baseline)
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

def predict_abc(AIG):
    # return 0
    if type(AIG) == str:
        return p_abc(AIG)
    else:
        return [p_abc(aig) for aig in AIG]

def p_abc(AIG):
    state = AIG.split('.')[0]
    libFile = LIBFILE
    logFile = 'tmp.log'

    circuitName, actions = state.split('_')
    circuitPath = os.path.join(BASEPATH, 'InitialAIG/test/' + circuitName + '.aig') # FIXME: train or test?
    
    aig_dir = os.path.join(BASEPATH, 'test_aig_files')
    
    if not os.path.exists(os.path.join(BASEPATH, 'test_aig_files', state + '.aig')):
        actionCmd = ''
        for action in actions:
            actionCmd += (synthesisOpToPosDic[int(action)] + '; ')
        initial_abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + actionCmd + "read_lib " + libFile + "; write " + AIG + "; print_stats\" > " + logFile
        os.system(initial_abcRunCmd) 
        os.system(f"mv {state}.aig {aig_dir}/")
    else:
        pass 
    
    abcRunCmd = "yosys-abc -c \"read " + f"{aig_dir}/{state}.aig" + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    #print(abcRunCmd)
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        adpVal = float(areaInformation[-9]) * float(areaInformation[-4])

    baseline = cal_baseline(AIG, circuitPath=circuitPath, libFile=libFile)
    finalVal = (baseline - adpVal) / baseline
    return finalVal

def clear_tmp_files():
    os.system("rm -rf ../task2/*.log")
    os.system("rm -rf ../task2/*.aig")
    os.system("rm -rf ../project/test_aig_files/*.aig")
    os.system("rm -rf aigFile/*.aig")
    os.system("rm -rf libFile/*.log")

from datetime import datetime

def search(AIG, method='greedy', maxsize=200, predict_fn=None, log_path=None, n_steps=10):
    os.makedirs('libFile', exist_ok=True)
    os.makedirs('aigFile', exist_ok=True)
    clear_tmp_files() # 删除 task2 文件夹和 project/test_aig_files 中的 .log 和 .aig 文件
    
    if not '_' in AIG:
        AIG = AIG.split('.')[0] + '_' + '.aig'
        
    search_process = Search(n_steps=n_steps, n_branch=7, predict_fn=predict_fn)
    AIG = search_process(AIG, method=method, maxsize=maxsize)

    actions = AIG.split('.')[0].split('_')[1]
    log_message(f'-------------------{AIG} results: -----------------------', log_path)
    log_message(f'Time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', log_path)
    bestVal = -float('inf')
    final_AIG_name = ''
    final_pred = 0
    for i in range(1, len(actions) + 1):
        ac = actions[:i]
        tmp_AIG = AIG.split('.')[0].split('_')[0] + '_' + ac + '.aig'
        
        log_message(f'{tmp_AIG}', log_path)
        tmp_pred = search_process.predict(tmp_AIG, future=False) # drop the future value !!!
        log_message(f'pred: {tmp_pred}', log_path)
        finalVal = predict_abc(tmp_AIG)
        log_message(f'gt: {finalVal}', log_path)
        if bestVal < finalVal:
            bestVal = finalVal
            final_pred = tmp_pred
            final_AIG_name = tmp_AIG
    log_message(f'Name: {final_AIG_name} final pred AIG: {final_pred} final gt AIG: {bestVal}', log_path)

import random

def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def test(AIG='alu4.aig', method='greedy', maxsize=200, predict_fn=None):
    os.makedirs('libFile', exist_ok=True)
    os.makedirs('aigFile', exist_ok=True)
    clear_tmp_files() # 删除 task2 文件夹和 project/test_aig_files 中的 .log 和 .aig 文件

    AIG = AIG
    if not '_' in AIG:
        AIG = AIG.split('.')[0] + '_' + '.aig'
    # if '.aig' in AIG:
    #     AIG = AIG.replace('.aig', '')
    # length = 40
    preds = []
    gts = []
    
    # search
    search_process = Search(n_steps=10, n_branch=7, predict_fn=predict_fn)
    log_message(f'AIG: {AIG} start')
    for i in range(2):
        cur = '' + str(random.randint(0, 6))
        for j in range(2):
            cur = cur + str(random.randint(0, 6))
            for z in range(2):
                cur = cur + str(random.randint(0, 6))
                for k in range(2):
                    cur = cur + str(random.randint(0, 6))
                    for l in range(2):
                        cur = cur + str(random.randint(0, 6))
                        # print(cur, 'here')
                        final_length = random.randint(1, 5)
                        new_cur = random.sample(cur, final_length)
                        # turn list into string
                        new_cur = ''.join(new_cur)
                        # print(new_cur)
                        new_AIG = AIG.split('.')[0] + new_cur + '.aig'
                        # print(new_AIG)
                        
                        pred_value = search_process.predict_fn(new_AIG, future=False)
                        preds.append(pred_value)
                        gt_value = predict_abc(new_AIG)
                        gts.append(gt_value)
                        # print('pred: ', pred_value, 'gt: ', gt_value)
                        log_message(f'{new_AIG} pred: {pred_value} gt: {gt_value}')
                        
                        # 移除最后一位
                        cur = cur[:-1]
                    cur = cur[:-1]
                cur = cur[:-1]
            cur = cur[:-1]
        cur = cur[:-1]
    
    # print(f'MSE = {np.mean((np.array(preds) - np.array(gts))**2)}')
    # print(f'MAE = {np.mean(np.abs(np.array(preds) - np.array(gts)))}')
    log_message(f'{AIG} MSE = {np.mean((np.array(preds) - np.array(gts))**2)}')
    log_message(f'{AIG} MAE = {np.mean(np.abs(np.array(preds) - np.array(gts)))}')
    # AIG = search_process(AIG, method=method, maxsize=maxsize)
    return preds, gts
    # print(AIG)
    # print('pred: ', search_process.predict_fn(AIG,future=False))     # pred: 0.26 final: -0.07
    
    # finalVal = predict_abc(AIG)
    # print('final: ', finalVal)

# test mse on test dataset
# ls_files = os.listdir('../project/InitialAIG/test')

# log_message('Testing...')
# for ls_fl in ls_files:
#     # search(ls_fl, method='BestFirstSearch', maxsize=24, predict_fn=None)    # NOTE
    
#     # ls_fl = 'mem_ctrl.aig'
#     if 'mem_ctrl' in ls_fl:
#         ls_fl = ls_fl.replace('mem_ctrl', 'memctrl')
#     # test
#     preds = []
#     gts = []
    
#     ps, gs = test(ls_fl, method='BestFirstSearch', maxsize=24, predict_fn=None)    # NOTE
#     preds.extend(ps)
#     gts.extend(gs)

# log_message(f'Final MSE = {np.mean((np.array(preds) - np.array(gts))**2)}')
# log_message(f'Final MAE = {np.mean(np.abs(np.array(preds) - np.array(gts)))}')
# clear_tmp_files()

import time
def args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Predict final test dataset')
    parser.add_argument('--method', type=str, default='greedy', help='search method', choices=['greedy', 'BestFirstSearch', 'DFS', 'BFS'])
    parser.add_argument('--maxsize', type=int, default=200, help='maxsize only for the BestFirstSearch method')
    parser.add_argument('--predict', type=str, default='abc_now', help='predict method', choices=['abc_now', 'abc_now_gnn_future', 'gnn_now_gnn_future'])
    parser.add_argument('--n_steps', type=int, default=10, help='n_steps for the search process')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    log_path = f'./log_test_dataset/{args.predict}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_path = f'{log_path}/method_{args.method}_step_{args.n_steps}_maxsize_{args.maxsize}_{current_time}.txt'
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    log_message(f'args: {args}', log_path)
    ls_files = os.listdir('../project/InitialAIG/test')
    if args.method == 'BestFirstSearch':
        assert args.maxsize <= 24, 'BestFirstSearch only support maxsize <= 24' # ??? NOTE

    for ls_fl in ls_files:
        time_start = time.time()
        if 'mem_ctrl' in ls_fl:
            ls_fl = ls_fl.replace('mem_ctrl', 'memctrl')
            
        if args.predict == 'abc_now':
            # abc now means only use the abc evaluation
            search(ls_fl, method=args.method, maxsize=args.maxsize, predict_fn=predict_abc, log_path=log_path, n_steps=args.n_steps)
        elif args.predict == 'gnn_now_gnn_future':
            search(ls_fl, method=args.method, maxsize=args.maxsize, predict_fn=None, log_path=log_path, n_steps=args.n_steps)    # NOTE
        else:
            raise NotImplementedError
        log_message(f'Time cost: {time.time() - time_start:.2f} s', log_path)
        
    clear_tmp_files()
    
# CUDA_VISIBLE_DEVICES=0 python task2.py   
# 2-5 s 一个 step -> 
# depth=5 19607 steps
# depth=4 2800 steps
