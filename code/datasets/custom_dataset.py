from torch.utils.data import Dataset, IterableDataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from utils import *


class InMemoryCSVDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        fe_name,
        file_name,
        feature_path,
        nb_samples=None,
        skip_rows=0,
    ):
        self.feature_path =feature_path
        self.dataset_name=dataset_name
        self.file_name=file_name
        self.fe_name=fe_name
        
        self.name = f"{dataset_name}/{fe_name}/{file_name}"
        self.data = pd.read_csv(
            self.feature_path,
            skiprows= 1  # +1, since we skip the header
            + skip_rows,
            nrows=nb_samples,
        )

    def __getitem__(self, index):
        if index > self.len:
            raise IndexError("EOF")

        return self.data[index].values, self.file_name

    def __len__(self):
        return len(self.data)
    
class IterativeCSVDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        fe_name,
        file_name,
        feature_path,
        nb_samples=None,
        chunksize=2**14,
        skip_rows=0,
    ):
        self.feature_path =feature_path
        self.dataset_name=dataset_name
        self.file_name=file_name
        self.fe_name=fe_name
        
        self.name = f"{dataset_name}/{fe_name}/{file_name}"
        self.data = pd.read_csv(
            self.feature_path,
            skiprows= 1  # +1, since we skip the header
            + skip_rows,
            chunksize=chunksize,
            nrows=nb_samples,
        )
        self.nb_samples=nb_samples
        self.skip_rows=skip_rows
        self.chunksize=chunksize
        self.skip_rows=skip_rows 
    
    def reset(self):
        self.data = pd.read_csv(
            self.feature_path,
            skiprows= 1  # +1, since we skip the header
            + self.skip_rows,
            chunksize=self.chunksize,
            nrows=self.nb_samples,
        )
        
    def __iter__(self):
        for chunk in self.data:
            for i in chunk.values:
                yield i, self.file_name
        
    def __len__(self):
        return self.nb_samples
    



def load_dataset(dataset_name, fe_name, file_name,
                 percentage=[0,1], batch_size=1024):
    dataset_info = load_dataset_info()
    
    dataset=dataset_info[dataset_name][fe_name][file_name]
    total_rows = dataset["num_rows"]
    feature_path = dataset["feature_path"]
    nb_samples=int(total_rows*(percentage[1]-percentage[0]))
    
    if batch_size is None:
        batch_size=nb_samples
        dataset_type=InMemoryCSVDataset
    else:
        dataset_type=IterativeCSVDataset
        
    return DataLoader(dataset_type(
            dataset_name,
            fe_name,
            file_name,
            feature_path, 
            nb_samples=nb_samples, 
            skip_rows=int(total_rows * percentage[0]),
        ), batch_size=batch_size)
