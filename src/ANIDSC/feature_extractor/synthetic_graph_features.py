import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..utils import *

class Node:
    def __init__(self, id, mean, std, dim):
        self.id=id
        self.mean=mean 
        self.std=std
        self.dim=dim 

        self.rng=np.random.default_rng()

    def gen_random_feature(self):
        return self.rng.normal(self.mean, self.std, size=self.dim)
    
    def gen_correlated_feature(self):
        val=self.rng.normal(self.mean, self.std)
        feature=[val]
        for i in range(self.dim-1):
            feature.append(val*i)
        
        return np.array(feature)

class Graph:
    def __init__(self, means, stds, node_dim, connections, output_file,meta_file):
        self.node_dim=node_dim
        
        self.node_list=[Node(i, j, k, node_dim) for i, (j, k) in enumerate(zip(means, stds))]

        self.connections=np.array(connections)
        self.rng=np.random.default_rng()
        
        feature_file = Path(output_file)
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        self.feature_file=open(feature_file, "w")
        
        meta_file = Path(meta_file)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file=open(meta_file, "w")

        self.drifts=["change_mean"] #"remove_edge","add_edge","alter_edge","new_node", "change_std"

    def simulate(self, num_drifts):
        count=0
        #initial state for random duration
        duration=self.rng.integers(5000,10000)
        
        
        self.meta_file.write(f"initial duration {duration}\n")
        count+=self.gen_connections(duration)

        #randomly choose and apply drifts
        for i in tqdm(range(num_drifts)):
            desc=self.apply_drift()
            duration=self.rng.integers(5000,10000)
            count+=self.gen_connections(duration)
            self.meta_file.write(f"{desc} {duration}\n")
        return count 
    
    def gen_connections(self, duration):
        feature_list=[]
        for t in range(duration):
            for i,j in self.connections.T:
                src=self.node_list[i]
                dst=self.node_list[j]
                feature_list.append(np.hstack([1, src.id, dst.id, 0, src.gen_correlated_feature(), dst.gen_correlated_feature(), np.zeros(35)]))
        
        np.savetxt(
            self.feature_file,
            np.vstack(feature_list),
            delimiter=",",
            fmt="%.7f"
        )
        return len(feature_list)
    
    def apply_drift(self):
        drift=self.rng.choice(self.drifts)
        desc=getattr(self, drift)()
        return desc

    def rand_mean(self):
        return self.rng.uniform(15,30)

    def rand_std(self):
        return self.rng.uniform(1,10)

    def change_mean(self):
        random_node=  self.node_list[1]#self.rand_node()
        random_node.mean= 30 #self.rand_mean()
        return f"alter {random_node.id} mean = {random_node.mean:.2f}"
    
    def change_std(self):
        random_node=self.rand_node()
        random_node.std=self.rand_std()
        return f"alter {random_node.id} std = {random_node.std:.2f}"

    def rand_node(self):
        return self.rng.choice(self.node_list)

    def new_node(self):
        rand_dst=self.rand_node()
        new_node=Node(len(self.node_list), self.rand_mean(), self.rand_std(), self.node_dim)
        self.node_list.append(new_node)
        self.connections=np.hstack([self.connections, [[rand_dst.id],[new_node.id]]])
        return f"new node {new_node.id} mean = {new_node.mean:.2f}, std = {new_node.std:.2f}, connected to {rand_dst.id}"
    
    def rand_edge(self):
        return self.rng.choice(self.connections.shape[1])

    def remove_edge(self):
        rand_edge_idx=self.rand_edge()
        rand_edge=self.connections[:, rand_edge_idx]
        self.connections=np.delete(self.connections, rand_edge_idx, 1)
        return f"removed {rand_edge}"

    def add_edge(self):
        src=self.rand_node().id
        dst=self.rand_node().id

        while ((self.connections[0]==src) & (self.connections[1]==dst)).any():
            src=self.rand_node().id
            dst=self.rand_node().id
        
        self.connections=np.hstack([self.connections, [[src],[dst]]])
        return f"add edge {src} {dst}"

    def alter_edge(self):
        removed_desc=self.remove_edge()
        add_desc=self.add_edge()
        return removed_desc+" "+add_desc
    
class SyntheticFeatureExtractor:
    def __init__(self, dataset_name, file_name):
        self.name="SyntheticFeatureExtractor"
        self.dataset_name=dataset_name
        self.file_name=file_name
        
        self.feature_file=f"../datasets/{dataset_name}/{self.name}/{file_name}.csv"
        self.meta_file=f"../datasets/{dataset_name}/{self.name}/{file_name}_meta.csv"
        self.g=Graph(means=[1,3,5,10],
        stds=[1,1,1,1],
        node_dim=15,
        connections=[[0,1,1,3],[2,2,3,2]],
        output_file=self.feature_file,
        meta_file=self.meta_file)

    def generate_features(self, n):
        self.count=self.g.simulate(n)
        
        # save file information
        data_info = load_dataset_info()

        if self.dataset_name not in data_info.keys():
            data_info[self.dataset_name] = {}

        if self.name not in data_info[self.dataset_name].keys():
            data_info[self.dataset_name][self.name] = {}

        data_info[self.dataset_name][self.name][self.file_name] = {
            "pcap_path": None,
            "feature_path": self.feature_file,
            "meta_path": self.meta_file,
            "num_rows": int(self.count),
        }

        save_dataset_info(data_info)
        print(
            f"written: {self.count}"
        )
