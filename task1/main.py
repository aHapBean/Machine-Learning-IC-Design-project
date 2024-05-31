import os
import numpy as np
import torch
from model import GCN
from dataset import get_dataset
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
    for data in tqdm(dataloader, desc='Training dataset'):
        data = data.to(device)
        optimizer.zero_grad()
    
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test(model, device, dataloader, criterion):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            mse = criterion(output, data.y).item()
            mae = torch.abs(output - data.y).mean().item()
            total_mse += mse
            total_mae += mae
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    return avg_mse, avg_mae

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Load dataset...')
    dataset = get_dataset(args.data)
    
    print('Load over !')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GCN(num_node_features=1).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss() 

    for epoch in range(args.max_epoch):  
        time_start = time.time()
        train_loss = train(model, device, train_loader, optimizer, criterion)
        mse, mae = test(model, device, test_loader, criterion=criterion)
        print(f'Time: {time.time() - time_start:.2f} Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae: .4f}')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../project/project_data', help='The path of data in task 1')
    parser.add_argument('--num-node-features', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = args_parser()
    main(args)
