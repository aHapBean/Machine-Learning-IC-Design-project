# Importing required libraries
import os
import re
import numpy as np
import torch
import abc_py
from torch_geometric.data import Data
from model import GCN

base_path = '../project/'

# Define the initial state and extract the circuit name and actions performed
state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = os.path.join(base_path, 'InitialAIG/train/' + circuitName + '.aig')
# print(circuitPath)

libFile = os.path.join(base_path, 'lib/7nm/7nm.lib')

if not os.path.exists(circuitPath) or not os.path.exists(libFile):
    raise ValueError('path error')

logFile = 'alu2.log'
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

# Execute the synthesis command
# os.system(abcRunCmd)
# print(abcRunCmd)

# raise ValueError("The code below is not executed")

# Function to evaluate the AIG with Yosys
def evaluate_aig(AIG, libFile, logFile):
    abcRunCmd = "yosys-abc -c \"read " + AIG + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    print(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        eval = float(areaInformation[-9]) * float(areaInformation[-4])
    return eval

print('CircuitPath ', circuitPath)
evaluate_score = evaluate_aig(circuitPath, libFile, logFile)    # NOTE FIXME 这个应该不是传入circuitPath，而是对应的state.aig ！！！！
print(evaluate_aig(circuitPath, libFile, logFile))

# Regularizing the evaluation using resyn2
def regularize_aig(eval):
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + "read_lib " + libFile + "; write " + nextState + "; write_bench -l " + nextState + "; map; topo; stime\" > " + logFile
    # abcRunCmd = "yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + "read_lib " + libFile + ";" + " write_bench -l " + nextState + "; map; topo; stime\" > " + logFile
    # print(nextState)
    # print(f'abcRunCmd: {abcRunCmd}')
    
    """
    alu2_0130622.aig
    abcRunCmd: yosys-abc -c "read ../project/InitialAIG/train/alu2.aig; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;read_lib ../project/lib/7nm/7nm.lib; write alu2_0130622.aig; write_bench -l alu2_0130622.aig; map; topo; stime" > alu2.log
    """
    os.system(abcRunCmd)
    # raise ValueError
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
        eval = 1 - eval / baseline
        return eval

print(f'normalized: {regularize_aig(evaluate_score)}')  # NOTE
raise ValueError
# raise ValueError
# Building the AIG representation through node connectivity

os.system(initial_abcRunCmd)    # 注意这个要在上面这两个之后

_abc = abc_py.AbcInterface()
# abcpy.
_abc.start()
print(state + '.aig')
# _abc.read(state + '.aig')
_abc.read(state + '.aig')
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
        edge_target_index.append(fanin)
data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
data['node_type'] = torch.tensor(data['node_type'], dtype=torch.double) # NOTE float
data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
data['nodes'] = numNodes

print(data)

node_features = data['node_type']
edge_index = data['edge_index']

graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(regularize_aig(evaluate_score)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=1).to(device)
graph_data = graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out, graph_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    
    
def test():
    model.eval()
    with torch.no_grad():
        logits = model(graph_data)
        test_mask = graph_data.test_mask
        pred = logits[test_mask].max(1)[1]
        correct = pred.eq(graph_data.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
    return acc

accuracy = test()
print(f'Test Accuracy: {accuracy:.4f}')
