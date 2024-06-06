# NOTE 

CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method DFS

CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method BFS

# 需要注意手动更改测试集的mem_ctrl的名字
        # if 'mem_ctrl' in ls_fl:
        #     ls_fl = ls_fl.replace('mem_ctrl', 'memctrl')