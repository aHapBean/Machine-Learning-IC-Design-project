CUDA_VISIBLE_DEVICES=3 python main.py

# 01

# CUDA_VISIBLE_DEVICES=1 python main.py --datasize all


# 00 注意这个模型其实是hybrid
CUDA_VISIBLE_DEVICES=3 python main.py --datasize 5000 --batch-size 32 --model EnhancedGCN --lr 0.001

# 01
# CUDA_VISIBLE_DEVICES=1 python main.py --datasize 1000
CUDA_VISIBLE_DEVICES=0 python main.py --datasize 5000 --batch-size 64 --model DeeperEnhancedGCN --lr 0.008

0.0001 lr有效


# 02
# CUDA_VISIBLE_DEVICES=2 python main.py --datasize 5000
CUDA_VISIBLE_DEVICES=2 python main.py --datasize 10000

# 03
# CUDA_VISIBLE_DEVICES=0 python main.py --datasize 1000 --lr 0.0001
CUDA_VISIBLE_DEVICES=0 python main.py --datasize 5000 --batch-size 64 --model PureGAT --lr 0.008 



# NOTE the lr
CUDA_VISIBLE_DEVICES=1 python main.py --datasize 5000 --batch-size 32 --model EnhancedGCN --lr 0.001
# NOTE the lr
CUDA_VISIBLE_DEVICES=0 python main.py --datasize 5000 --batch-size 64 --model DeeperEnhancedGCN --lr 0.008

# 