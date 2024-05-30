import torch.utils.data as Data
import pickle 
import os
from tqdm import tqdm

class dataset(Data.Dataset):
    def __init__(self, path):
        data_ls = os.listdir(path)
        self.aig_names = []
        self.labels = []
        
        cnt = 0
        for data_file in tqdm(data_ls, desc="Loading data"):  # Use tqdm for progress bar
            cnt += 1
            if cnt >= 100:
                break
            data_path = os.path.join(path, data_file)
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            aig_name = data['input']
            label = data['target']
            self.aig_names.extend(aig_name)
            self.labels.extend(label)
            
        # raise NotImplementedError
    
    def __getitem__(self, index):
        return self.aig_names[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def get_dataset(path):
    return dataset(path)

if __name__ == '__main__':
    get_dataset('../project/project_data')