# CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 3 --method DFS --predict gnn_now
# 01
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --method greedy --predict gnn_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 10 --maxsize 25 --method BestFirstSearch --predict gnn_now
CUDA_VISIBLE_DEVICES=0 python task2.py --n_steps 4 --method BFS --predict gnn_now

