# NOTE 

CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method DFS

CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method BFS

# 需要注意手动更改测试集的mem_ctrl的名字
        # if 'mem_ctrl' in ls_fl:
        #     ls_fl = ls_fl.replace('mem_ctrl', 'memctrl')


# TODO 计划结果
abc_now
DFS (depth=4) BFS (depth=4) Greedy BestFS

abc_now_gnn_future
DFS (depth=4) BFS (depth=4) Greedy BestFS

gnn_now_gnn_future
DFS (depth=5) BFS (depth=5) Greedy BestFS ?