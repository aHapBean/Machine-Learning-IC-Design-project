# CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 3 --method DFS --predict gnn_now_gnn_future
# 01
CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 10 --method greedy --predict gnn_now_gnn_future --finetuned
CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 10 --maxsize 10 --method BestFirstSearch --predict gnn_now_gnn_future --finetuned
CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 4 --method BFS --predict gnn_now_gnn_future --finetuned
CUDA_VISIBLE_DEVICES=1 python task2.py --n_steps 3 --method BFS --predict gnn_now_gnn_future --finetuned

