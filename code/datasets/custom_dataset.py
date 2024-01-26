import torch
import pandas as pd
from pathlib import Path
import numpy as np
from utils import *


class IterativeCSVDataset(torch.utils.data.Dataset):
    def __init__(self, name, feature_path, nb_samples, label_path=None, chunksize=1024, skip_rows=0):
        self.feature_path = feature_path
        self.file_name=Path(feature_path).stem 
        self.label_path=label_path
        self.name=name
        #if chunksize is none, load all data in single batch
        if chunksize is None:
            chunksize=nb_samples
            self.len=0
        else:
            self.len = nb_samples // chunksize
                        
        self.chunksize = chunksize        
        self.skip_rows=skip_rows

    def __getitem__(self, index):
        
        if index > self.len:
            raise IndexError("EOF")
        
        x = pd.read_csv(
                self.feature_path,
                skiprows=index * self.chunksize + 1 + self.skip_rows,  #+1, since we skip the header
                nrows=self.chunksize)
        x = x.values
        
        if self.label_path is None:
            y=np.full(x.shape[0], self.file_name)
        else:
            y = pd.read_csv(
                    self.label_path,
                    skiprows=index * self.chunksize + 1 + self.skip_rows,  #+1, since we skip the header
                    nrows=self.chunksize)
            y = y.values
        
        return x, y

    def __len__(self):
        return self.len
    
    
def load_dataset(dataset_id, train_val_test=[0.8,0.1,0.1], batch_size=1024):
    dataset_info=load_dataset_info()
    total_rows=dataset_info[dataset_id]["num_rows"]
    feature_path=dataset_info[dataset_id]["feature_path"]
    
    if train_val_test == False:
        return IterativeCSVDataset(dataset_id, feature_path, total_rows, chunksize=batch_size)

    train_dataset=IterativeCSVDataset(dataset_id+"_train",feature_path, int(total_rows*train_val_test[0]), chunksize=batch_size)
    val_dataset=IterativeCSVDataset(dataset_id+"_val",feature_path, int(total_rows*train_val_test[1]), chunksize=batch_size, skip_rows=int(total_rows*train_val_test[0]))
    test_dataset=IterativeCSVDataset(dataset_id+"_test",feature_path, int(total_rows*train_val_test[2]), chunksize=batch_size, skip_rows=int(total_rows*sum(train_val_test[:2])))
    
    return {"train":train_dataset, "test":test_dataset, "val":val_dataset}
    
        