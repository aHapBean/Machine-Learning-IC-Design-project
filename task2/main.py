import os
import numpy as np
import torch
from model import GCN, EnhancedGCN, DeeperEnhancedGCN, PureGAT, GIN
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
    log_path = './log_task2'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = f'{log_path}/{args.model}_size_{args.datasize}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = f'{log_path}/log_{args.model}_{args.datasize}_size_{args.lr}_lr_{args.batch_size}_bs_{current_time}.txt'
    log_message(str(args), log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Load dataset...')
    dataset = get_dataset(args.data, args.datasize)
    
    print('Load over !')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # NOTE 伪随机
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'Train length {len(train_dataset)}, Test length {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model = GCN(num_node_features=2).to(device)     # NOTE 2
    if args.model == 'GCN':
        model = GCN(num_node_features=2).to(device)
    elif args.model == 'EnhancedGCN':
        model = EnhancedGCN(num_node_features=2).to(device)
    elif args.model == 'DeeperEnhancedGCN':
        model = DeeperEnhancedGCN(num_node_features=2).to(device)
    elif args.model == 'PureGAT':
        model = PureGAT(num_node_features=2).to(device)
    elif args.model == 'GIN':
        model = GIN(num_node_features=2).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss() 

    best_mse = float('inf')
    best_mae = float('inf')
    for epoch in range(args.max_epoch):  
        time_start = time.time()
        train_loss = train(model, device, train_loader, optimizer, criterion)
        mse, mae = test(model, device, test_loader, criterion=criterion)
        # mse, mae = 0, 0
        if mse < best_mse:
            best_mse = mse
            ls_file = os.listdir(log_path)
            flag = False 
            for fl in ls_file:
                if fl.endswith('.pth') and 'best_mse' in fl:
                    flag = True
                    tmp_best = fl.replace('.pth', '').split('_')[-1]
                    # tmp_best = float(fl.split('_')[-1].split('.')[0])
                    if best_mse < float(tmp_best):
                        os.remove(f'{log_path}/{fl}')
                        torch.save(model.state_dict(), f'{log_path}/best_mse_{best_mse:.5f}.pth')
                    break
            if not flag:
                torch.save(model.state_dict(), f'{log_path}/best_mse_{best_mse:.5f}.pth')
        
        if mae < best_mae:  # NOTE here seperate MSE and MAE
            best_mae = mae
            ls_file = os.listdir(log_path)
            for fl in ls_file:
                flag = False
                if fl.endswith('.pth') and 'best_mae' in fl:
                    flag = True
                    tmp_best = fl.replace('.pth', '').split('_')[-1]
                    # tmp_best = float(fl.split('_')[-1].split('.')[0])
                    if best_mae < float(tmp_best):
                        os.remove(f'{log_path}/{fl}')
                        torch.save(model.state_dict(), f'{log_path}/best_mae_{best_mae:.5f}.pth')
                    break
            if not flag:
                torch.save(model.state_dict(), f'{log_path}/best_mae_{best_mae:.5f}.pth')
            
        log_message(f'Time: {time.time() - time_start:.2f} Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test MSE: {mse:.4f}, Best MSE: {best_mse:.4f}, Test MAE: {mae: .4f}, Best MAE: {best_mae:.4f}', log_file)
        

def test_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Load dataset...')
    dataset = get_dataset(args.data, args.datasize, task=args.task)
    
    print('Load over !')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # NOTE 伪随机
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'Train length {len(train_dataset)}, Test length {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    log_path = './log_task2'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = f'{log_path}/{args.model}_size_{args.datasize}'
    criterion = torch.nn.MSELoss() 
    
    if args.model == 'GCN':
        model = GCN(num_node_features=2).to(device)
    elif args.model == 'EnhancedGCN':
        model = EnhancedGCN(num_node_features=2).to(device)
    elif args.model == 'DeeperEnhancedGCN':
        model = DeeperEnhancedGCN(num_node_features=2).to(device)
    elif args.model == 'PureGAT':
        model = PureGAT(num_node_features=2).to(device)
    elif args.model == 'GIN':
        model = GIN(num_node_features=2).to(device)
    else:
        raise NotImplementedError
    
    flag = False
    ls_file = os.listdir(log_path)
    for ls in ls_file:
        if ls.endswith('.pth') and 'best_mae' in ls:
            flag = True
            model.load_state_dict(torch.load(os.path.join(log_path, ls)))
            break
    if not flag:
        raise ValueError('No best model found !!!')
    
    for epoch in range(1):  
        time_start = time.time()
        mse, mae = test(model, device, test_loader, criterion=criterion, test=args.test)
        print(f'Time: {time.time() - time_start:.2f} [Test MSE]: {mse:.4f}, [Test MAE]: {mae: .4f}')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../project/project_data2', help='The path of data in task 2')
    # parser.add_argument('--num-node-features', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--datasize', type=str, default='100')    # 使用的数据集大小
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', default='EnhancedGCN', help='The model to use')
    parser.add_argument('--test', action='store_true', help='Test the model', default=False)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = args_parser()
    assert args.task == 1
    if args.test:
        print('Test the model...')
        test_main(args)
    else:
        main(args)