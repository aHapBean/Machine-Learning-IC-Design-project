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
# 00
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method DFS --predict abc_now
# 01
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method BFS --predict abc_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --method greedy --predict abc_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --maxsize 10 --method BestFirstSearch --predict abc_now
# 可以尝试 25 50 100 等等

CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method DFS --predict gnn_now
# 01
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method BFS --predict gnn_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --method greedy --predict gnn_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --maxsize 10 --method BestFirstSearch --predict gnn_now



CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method DFS --predict abc_now_gnn_future
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method BFS --predict abc_now_gnn_future

# 02
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --method greedy --predict abc_now_gnn_future
# 03
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --maxsize 25 --method BestFirstSearch --predict abc_now_gnn_future



# 初步判断有一定效果，比如apex结果比log_testing更好
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method DFS --predict gnn_now_gnn_future
# 
CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 4 --method BFS --predict gnn_now_gnn_future
# 04 TODO will running
# CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method BFS --predict gnn_now_gnn_future

# 4
CUDA_VISIBLE_DEVICES=2 python task2.py --n_steps 10 --method greedy --predict gnn_now_gnn_future
CUDA_VISIBLE_DEVICES=2 python task2.py --n_steps 10 --maxsize 25 --method BestFirstSearch --predict gnn_now_gnn_future


# TODO searchRandom
python task2.py --maxsize 500 --method RandomSearch --predict abc_now

# 可以再跑跑 n_steps = 3 的结果 DFS BFS

# 可以再加一组 random 的结果
??????


finetune的结果
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method DFS --predict gnn_now --finetuned
# 01
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method BFS --predict gnn_now --finetuned
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --method greedy --predict gnn_now --finetuned
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --maxsize 10 --method BestFirstSearch --predict gnn_now --finetuned