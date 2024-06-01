import os
import numpy as np
import torch
from model import GCN, EnhancedGCN
from dataset import get_dataset
from torch_geometric.data import Data
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import abc_py
import time

from tqdm import tqdm 
def train(model, device, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    # for data in dataset:
    for data in tqdm(dataloader, desc='Training dataset', leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        # print(lbl)
    
        output = model(data)
        # print(graph_data.y.shape, ' ', output.shape)
        # print(graph_data.y)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test(model, device, dataloader, criterion):
    model.eval()
    total_mse = 0
    total_mae = 0
    
    cnt = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            mse = criterion(output, data.y).item()
            mae = torch.abs(output - data.y).mean().item()
            total_mse += mse
            total_mae += mae
            cnt += 1
            if cnt < 6:
                print(f'pred: {output.item()}, label: {data.y.item()}')
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    return avg_mse, avg_mae

def log_message(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

import datetime 
def main(args):
    # log file
    log_path = './log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = f'{log_path}/log_{args.datasize}_size_{args.lr}_lr_{current_time}.txt'
    log_message(str(args), log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Load dataset...')
    dataset = get_dataset(args.data, args.datasize)
    
    print('Load over !')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model = GCN(num_node_features=2).to(device)     # NOTE 2
    model = EnhancedGCN(num_node_features=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss() 

    best_mse = float('inf')
    best_mae = float('inf')
    for epoch in range(args.max_epoch):  
        time_start = time.time()
        train_loss = train(model, device, train_loader, optimizer, criterion)
        mse, mae = test(model, device, test_loader, criterion=criterion)
        if mse < best_mse:
            best_mse = mse
        
        if mae < best_mae:  # NOTE here seperate MSE and MAE
            best_mae = mae
            
        log_message(f'Time: {time.time() - time_start:.2f} Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test MSE: {mse:.4f}, Best MSE: {best_mse:.4f}, Test MAE: {mae: .4f}, Best MAE: {best_mae:.4f}', log_file)
        

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../project/project_data', help='The path of data in task 1')
    # parser.add_argument('--num-node-features', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--datasize', type=str, default='100')    # 使用的数据集大小
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = args_parser()
    main(args)
