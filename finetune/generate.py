import numpy as np
import os 
import abc_py
import re 

def generate():
    ret = ''
    for i in range(10):
        ret += str(np.random.randint(0, 7)) # [0, 6]
    return ret


RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"

def cal_baseline(AIG, libFile='../project/lib/7nm/7nm.lib'):
    """根据 InitialAIG 里面的文件来获取 AIG 的 baseline"""
    state = AIG.split('.')[0]
    randomID = np.random.randint(100000)
    randomID2 = np.random.randint(100000)
    
    logFile = os.path.join('libFile', state + str(randomID) + '_' + str(randomID2) + 'tmp.log')
    nextState = os.path.join('aigFile', AIG)  
    if not os.path.exists(os.path.join('aigFile')):
        os.makedirs(os.path.join('aigFile'))
    if not os.path.exists(os.path.join('libFile')):
        os.makedirs(os.path.join('libFile'))

    circuitPath = os.path.join('../project/test_aig_files/' + AIG)

    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + "read_lib " + libFile + "; write " + nextState + "; write_bench -l " + nextState + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    # yosys-abc -c "read ../project/InitialAIG/test/c880.aig; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;read_lib ../project/lib/7nm/7nm.lib; write aigFile/c880_5243.aig; write_bench -l aigFile/c88      "ubun" 21:11 06-6月-240_5243.aig; map; topo; stime" > libFile/c880_5243tmp.log
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    #print("baseline:", baseline)
    
    os.system(f"rm {logFile}")
    os.system(f'rm {nextState}')
    return baseline


def evaluate_AIG(AIG):
    """根据 InitialAIG 里面的文件来获取 AIG 的 regularized score"""
    # AIG: c880_123.aig
    assert AIG.endswith('.aig')
    state = AIG.split('.')[0]
    circuitName, actions = state.split('_')
    
    libFile='../project/lib/7nm/7nm.lib'
    circuitPath = os.path.join('../project/test_aig_files/' + AIG)   # the original AIG file
    # print(circuitPath)
    
    randomID = np.random.randint(100000)
    randomID2 = np.random.randint(100000)
    logFile = os.path.join('libFile', state + str(randomID) + '_' + str(randomID2) + 'tmp.log')
    if not os.path.exists(os.path.join('libFile')):
        os.makedirs(os.path.join('libFile'))
        
    # yosys-abc -c "read ../project/InitialAIG/train/alu2.aig; read_lib ../project/lib/7nm/7nm.lib; map; topo; stime" > alu2.log
    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    
    # FIXME NOT the name of Log File !!!
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        eval = float(areaInformation[-9]) * float(areaInformation[-4])

    baseline = cal_baseline(AIG)
    regularized_eval = 1 - eval / baseline
    # print("eval:",regularized_eval)
    os.system(f"rm {logFile}")
    # raise ValueError
    return regularized_eval

def calculate_AIG(state):
    """
    input alu_0123
    
    """
    base_path = '../project/'
    circuitName, actions = state.split('_')
    circuitPath = os.path.join(base_path, 'InitialAIG/test/' + circuitName + '.aig')   # NOTE only train dir ??应该是, test应该是用来设计aig的

    if not os.path.exists(os.path.join(base_path, 'test_aig_files', state + '.aig')):
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
        if not os.path.exists(os.path.join(base_path, 'test_aig_files/')):
            os.makedirs(os.path.join(base_path, 'test_aig_files/'))
        os.system(f"mv {state}.aig {os.path.join(base_path, 'test_aig_files/')}")
        # raise ValueError
        # print(f"path {os.path.join(base_path, 'test_aig_files/')}")
    else:
        pass
    
    return evaluate_AIG(state + '.aig')

from tqdm import tqdm 
import pickle 

def main():
    base_path = '../project/InitialAIG/test'
    fls = os.listdir(base_path)
    print(len(fls)) # 20
    # 20 * 50 * 10 = 10000
    
    save_dir = '../project/project_finetune_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cases_per_aig = 50
    
    for fl in tqdm(fls, desc='AIG files'):
        assert fl.endswith('.aig')
        # print(fl)
        cnt = 0
        for _ in range(cases_per_aig):
            ls_input = []
            ls_target = []
            
            seq = generate()
            prev = fl.split('.')[0] + '_'
            ls_input.append(prev)
            ls_target.append(calculate_AIG(ls_input[-1]))
            
            for i in range(len(seq)):
                ls_input.append(prev + seq[:i+1])   # NOTE no .aig !!
                ls_target.append(calculate_AIG(ls_input[-1]))
            
            ls_dict = {}
            ls_dict['input'] = ls_input
            ls_dict['target'] = ls_target
            
            # 保存ls_dict为pkl文件，名字为 prev + '_' + str(cnt) + '.pkl'
            name = prev + '_' + str(cnt) + '.pkl'
            with open(os.path.join(save_dir, name), 'wb') as f:
                pickle.dump(ls_dict, f)
            
            # with open(os.path.join(save_dir, name), 'rb') as f:
            #     data = pickle.load(f)
            # print(data)
            # raise ValueError
            
            cnt += 1
            
    
    print(ls_input)

    
    
    
main()

"""
{'input': ['adder_', 'adder_4', 'adder_42', 'adder_423', 'adder_4234', 'adder_42345', 'adder_423455', 'adder_4234552', 'adder_42345526', 'adder_423455260', 'adder_4234552604'], 
'target': [0.13195968288377902, 0.15156050835809987, 0.15156050835809987, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923]}
"""
