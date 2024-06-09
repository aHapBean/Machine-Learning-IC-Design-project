import pickle

# 读取.pkl文件
# with open('../project/project_data/alu2_0.pkl', 'rb') as f:
#     data = pickle.load(f)

with open('../project/project_data2/table5_9.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印内容
print(data)

"""
adder_0
{'input': ['adder_', 'adder_4', 'adder_42', 'adder_423', 'adder_4234', 'adder_42345', 'adder_423455', 'adder_4234552', 'adder_42345526', 'adder_423455260', 'adder_4234552604'], 
'target': [0.13195968288377902, 0.15156050835809987, 0.15156050835809987, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923]}

adder_1: 这是同一个板子的sequence变化
{'input': ['adder_', 'adder_4', 'adder_40', 'adder_405', 'adder_4052', 'adder_40524', 'adder_405243', 'adder_4052432', 'adder_40524323', 'adder_405243233', 'adder_4052432332'], 
'target': [0.13195968288377902, 0.15156050835809987, 0.15218239297905237, 0.15218239297905237, 0.15218239297905237, 0.15218239297905237, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923, 0.15951806352704923]}


alu2_0
{'input': ['alu2_', 'alu2_3', 'alu2_34', 'alu2_340', 'alu2_3403', 'alu2_34030', 'alu2_340300', 'alu2_3403004', 'alu2_34030046', 'alu2_340300464', 'alu2_3403004646'], 
'target': [0.04640426009302521, -0.14216200522287042, -0.020083823920083375, -0.03682304023361242, -0.10174034417146048, -0.10174034417146048, -0.10174034417146048, -0.054227023591464175, 0.012262824003044055, 0.012262824003044055, 0.05431800500299352]}
# 直接计算出来的normalized: 0.04640426009302523
"""


"""
project data2: 
{'input': ['table5_', 'table5_4', 'table5_41', 'table5_414', 'table5_4144', 'table5_41443', 'table5_414435', 'table5_4144351', 'table5_41443515', 'table5_414435156', 'table5_4144351561'], 
'target': [0.28965340279971313, 0.13256239152511862, 0.07765634435828728, 0.029418621337327575, 0.029418621337327575, 0.11688162185019511, 0.07979442799985055, 0.09160909728519852, 0.055324609914986195, -0.03667222202409462, 0.0]}

"""

