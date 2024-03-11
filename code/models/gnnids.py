import torch
import numpy as np
from .base_model import BaseOnlineODModel, TorchSaveMixin
from torch_geometric.data import Data
from collections import defaultdict
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn.models import GCN
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle 
from utils import *

def get_node_map(fe_name, dataset_name):
    with open(f"../../datasets/{dataset_name}/{fe_name}/state.pkl", "rb") as pf:
        state=pickle.load(pf)
    return state["node_map"]

def find_edge_indices(edges, srcID, dstID):
    search_edges=torch.vstack([srcID, dstID])
    column_indices = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        index = torch.nonzero(mask, as_tuple=False)
        if index.size(0) > 0:
            # If the column exists, append its index
            
            column_indices.append(index.item())
        else:
            # If the column does not exist, append None
            column_indices.append(-1)
    return torch.tensor(column_indices)


class GNNIDS(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, device='cuda',
                 n_features=15, preprocessors=[],
                 **kwargs):
        self.device=device
        
        BaseOnlineODModel.__init__(self,
            model_name='GNNIDS', n_features=n_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
    
        self.G=Data()
        
        self.processed=0
        
        self.encoder=GCN(in_channels=n_features, hidden_channels=8, out_channels=2, num_layers=3).to(self.device)
        self.decoder=GCN(in_channels=2, hidden_channels=8, out_channels=n_features, num_layers=3).to(self.device)
        
        self.loss=torch.nn.MSELoss(reduction='none').to(self.device)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        
        self.additional_params+=["G","processed"]
    
    def to_device(self, x):
        return x.float().to(self.device)
    
    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        unique_nodes=torch.unique(torch.cat([srcID, dstID]))
        max_idx=torch.max(unique_nodes)+1
        
        if self.G.x is None:
            expand_size=max_idx
            self.G.x=torch.empty((0, dst_feature.size(1))).to(self.device)
        else:
            expand_size=max_idx-self.G.x.size(0)
            
        
        if expand_size>0:
            self.G.x=torch.vstack([self.G.x, torch.zeros((expand_size, dst_feature.size(1))).to(self.device)]).to(self.device)
         
        for i in range(len(srcID)):
            self.G.x[srcID[i]]=src_feature[i]
            # if self.node_count[srcID[i]] > 1:
            #     self.G.x[srcID[i]]=(src_feature[i] - self.node_mean[srcID[i]])/torch.sqrt(self.node_var[srcID[i]]/self.node_count[srcID[i]])
            self.G.x[dstID[i]]=dst_feature[i]
            # if self.node_count[dstID[i]]>1:
            #     self.G.x[dstID[i]]=(dst_feature[i]- self.node_mean[dstID[i]])/torch.sqrt(self.node_var[dstID[i]]/self.node_count[dstID[i]])
           
        return unique_nodes
    
    def update_edges(self, srcID, dstID, edge_feature):
        
        # update all
        if self.G.edge_index is None:
            self.G.edge_index=torch.vstack([srcID,dstID])
            self.G.edge_attr=edge_feature
            return
        
        edge_indices=find_edge_indices(self.G.edge_index, srcID, dstID)
        
        existing_edge_idx= edge_indices != -1 
        
        #update found edges
        self.G.edge_attr[edge_indices[existing_edge_idx]]=edge_feature[existing_edge_idx]
        
        num_new_edges=torch.count_nonzero(~existing_edge_idx)
        if num_new_edges>0:
            new_edges=torch.vstack([srcID[~existing_edge_idx],dstID[~existing_edge_idx]])
            self.G.edge_index=torch.hstack([self.G.edge_index, new_edges])
            self.G.edge_attr=torch.vstack([self.G.edge_attr, edge_feature[~existing_edge_idx]])      
    
    def visualize_graph(self, dataset_name, fe_name, file_name):
        G = nx.DiGraph()
        G_sub=nx.DiGraph()
        
        edge_index = self.G.edge_index.clone().cpu().numpy()
        
        node_map=get_node_map(fe_name, dataset_name)
        node_map = dict((v,k) for k,v in node_map.items())
        
        G.add_edges_from(edge_index.T)
        G = nx.relabel_nodes(G, node_map)
        
        G_sub.add_edges_from(self.subgraph_edge_index.cpu().numpy().T)
        G_sub=nx.relabel_nodes(G_sub, node_map)
        
        fig, ax = plt.subplots(figsize=(8, 12), nrows=2)

        # Visualize the graph using NetworkX
        pos = nx.fruchterman_reingold_layout(G)  # Layout algorithm for positioning nodes
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, arrows=True, edge_color='gray', linewidths=0.5, font_size=10, ax=ax[0])
        ax[0].set_title(f'Network after {self.processed} packets')

        # Visualize the subgraph using NetworkX
        pos = nx.fruchterman_reingold_layout(G_sub)  # Layout algorithm for positioning nodes
        nx.draw(G_sub, pos, with_labels=True, node_color='skyblue', node_size=800, arrows=True, edge_color='gray', linewidths=0.5, font_size=10, ax=ax[1])
        ax[1].set_title(f'SubGraph after {self.processed} packets')
        # Save the image as a file (e.g., PNG)
        
        path=Path(f"../../datasets/{dataset_name}/{fe_name}/graph_plots/{file_name}/{self.model_name}_{self.processed}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format='png')

        # Close the figure to release resources
        plt.close(fig)
    
    def split_data(self, x):
        srcIDs=x[:, 1].long()
        dstIDs=x[:, 2].long()
        protocols=x[:, 3].long()
        
        src_features=x[:, 4:19]
        dst_features=x[:, 19:34]
        edge_features=x[:, 34:]
        
        return srcIDs, dstIDs, protocols, src_features, dst_features,edge_features
    
    def preprocess_nodes(self, IDs, features):
        unique_nodes=torch.unique(IDs)
        max_idx=torch.max(unique_nodes)+1
        
        if not hasattr(self, 'node_mean'):
            expand_size=max_idx
            self.node_mean=torch.empty((0, features.size(1))).to(self.device)
            self.node_var=torch.empty((0, features.size(1))).to(self.device)
            self.node_count=torch.empty(0).to(self.device)
        else:
            expand_size=max_idx - self.node_mean.size(0)
            
        if expand_size>0:
            self.node_mean=torch.vstack([self.node_mean, torch.zeros((expand_size, features.size(1))).to(self.device)]).to(self.device)
            self.node_var=torch.vstack([self.node_var, torch.zeros((expand_size, features.size(1))).to(self.device)]).to(self.device)
            self.node_count=torch.hstack([self.node_count, torch.zeros(expand_size).to(self.device)]).to(self.device)
        

        for i in unique_nodes:
            node_features=features[IDs==i]
            self.node_count[i]+=node_features.size(0)
            tmp_diff=node_features.mean()-self.node_mean[i]
            self.node_mean[i]=self.node_mean[i]+(node_features.mean()-self.node_mean[i])/self.node_count[i]
            self.node_var[i]=self.node_var[i]+(node_features.mean()-self.node_mean[i])*tmp_diff 
            
            
    def process(self, x):
        threshold=self.get_threshold()
        x=self.preprocess(x)
        x=x.to(self.device)
        
        
        diff = x[1:, 0] - x[:-1, 0]
        
        split_indices = torch.nonzero(diff, as_tuple=True)[1] + 1
       
        torch.split(x, split_indices, dim=1)
        update_mask=x[:,0]==1
        # Gather rows where the mask is True
        update_rows = x[update_mask]
        
        if update_rows.size(0)>0:
            srcIDs, dstIDs, protocols, src_features, dst_features,edge_features=self.split_data(update_rows)

            # self.preprocess_nodes(srcIDs, src_features)
            # self.preprocess_nodes(dstIDs, dst_features)
            
            #find index of last update
            _, indices=uniqueXT(update_rows[:,1:3], return_index=True, occur_last=True, dim=0)
            
            
            affected_nodes=self.update_nodes(srcIDs[indices], src_features[indices], dstIDs[indices], dst_features[indices])
            self.update_edges(srcIDs[indices], dstIDs[indices], edge_features[indices])
            self.processed+=update_rows.size(0)
        
        # delete links 
        delete_rows = x[~update_mask]
        if delete_rows.size(0)>0:
            #find indices of rows for deletion
            srcIDs, dstIDs, protocols, _, _, _=self.split_data(delete_rows)
            edge_indices=find_edge_indices(self.G.edge_index, srcIDs, dstIDs)
            edge_mask = torch.ones(self.G.edge_index.size(1), dtype=torch.bool)
            edge_mask[edge_indices] = False
            self.G.edge_index=self.G.edge_index[:,edge_mask]
            self.G.edge_attr=self.G.edge_attr[edge_mask]
        
        if update_rows.size(0)==0:
            return torch.tensor([0]),threshold
    
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(affected_nodes, num_hops=3,
                                                                edge_index=self.G.edge_index, 
                                                                num_nodes=self.G.x.size(0),
                                                                relabel_nodes=True)

        subgraph_nodes=self.G.x[subset].float()
        
        self.subgraph_nodes=subgraph_nodes
        self.subgraph_edge_index=edge_index
        
              
        self.optimizer.zero_grad()
        latent=self.encoder(subgraph_nodes, edge_index, edge_attr=self.G.edge_attr[edge_mask])
        recon=self.decoder(latent, edge_index, edge_attr=self.G.edge_attr[edge_mask])
        
        loss=self.loss(subgraph_nodes, recon).mean(1)
        anomaly_score=loss.clone().detach().cpu().numpy()
        
        loss=loss.mean()
        loss.backward()
        self.optimizer.step()
            
        self.score_hist.extend(anomaly_score)
        return anomaly_score, threshold
            
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.additional_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
                