import torch
from math import copysign,fabs,sqrt
import numpy as np
from .base_model import BaseOnlineODModel, TorchSaveMixin
from torch_geometric.data import Data, HeteroData
from collections import defaultdict
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GAT,GAE,MLP
from torch_geometric.utils import k_hop_subgraph,to_networkx, remove_isolated_nodes
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle 
from utils import *
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,TwoSlopeNorm
import matplotlib.patches as mpatches
import torch_geometric.transforms as T
from collections import deque
import scipy
from pytdigest import TDigest
torch.set_printoptions(precision=2)
class LivePercentile:
    def __init__(self, ndim=None):
        """ Constructs a LiveStream object
        """
        
        if isinstance(ndim, int):
            self.dims=[TDigest() for _ in range(ndim)]
            self.initialized=False
            self.ndim=ndim
        elif isinstance(ndim, list):
            self.dims=self.of_centroids(ndim)
            self.ndim=len(ndim)
            self.initialized=True
        else:
            raise ValueError("ndim must be int or list")

    def add(self, item):
        """ Adds another datum """
        item=item.cpu().numpy()
        for i, n in enumerate(item):
            self.dims[i].update(n)
        self.initialized=True

    def quantiles(self, p):
        """ Returns a list of tuples of the quantile and its location """
        
        if not self.initialized:
            return None 
        percentiles=np.zeros((len(p),self.ndim))
        
        for d in range(self.ndim):
            percentiles[:,d]=self.dims[d].inverse_cdf(p)
        
        return torch.tensor(percentiles).float()
    
    def to_centroids(self):
        return [i.get_centroids() for i in self.dims]
    
    def of_centroids(self, dim_list):
        return [TDigest.of_centroids(i) for i in dim_list]        

def generate_theta(n, dim):
    """generates n slices to be used to slice latent space, with dim dimensions. the slices are unit vectors"""
    rand=np.random.normal(size=[n, dim])
    theta= np.linalg.norm(rand, axis=1, keepdims=True)
    return torch.tensor(rand/theta)


def generate_z(n, dim, shape, center=0):
    """generates n samples with shape in dim dimensions, represents the prior distribution"""
    if shape == "uniform":
        z = 2 * np.random.uniform(size=[n, dim]) - 1
    elif shape == "circular":
        u = np.random.normal(size=[n, dim])
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        normalised_u=u/u_norm
        r = np.random.uniform(size=[n])**(1.0 / dim)
        z = np.expand_dims(r, axis=1) * normalised_u
        z += center
    return torch.tensor(z)


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        
    def forward(self, edges1, edges2, num_nodes):
        # Convert edges to adjacency matrices
        adj1 = self.edges_to_adjacency(edges1, num_nodes)
        adj2 = self.edges_to_adjacency(edges2, num_nodes)
        
        # Compute intersection and union
        intersection = torch.minimum(adj1, adj2)
        union = torch.maximum(adj1, adj2)
        
        return intersection.sum() / union.sum()

    def edges_to_adjacency(self, edges, num_nodes):
        adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
        adjacency[edges[0], edges[1]] = 1
        return adjacency

def get_node_map(fe_name, dataset_name):
    with open(f"../../datasets/{dataset_name}/{fe_name}/state.pkl", "rb") as pf:
        state=pickle.load(pf)
    return state["node_map"]

def get_protocol_map(fe_name, dataset_name):
    with open(f"../../datasets/{dataset_name}/{fe_name}/state.pkl", "rb") as pf:
        state=pickle.load(pf)
    return state["protocol_map"]

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

def edge_exists(edges, search_edges):
    label = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        label.append(mask.any())
    return torch.tensor(label)

def complete_directed_graph_edges(n):
    # Create an array of node indices
    nodes = torch.arange(n)
    
    # Generate all possible pairs of nodes
    pairs = torch.cartesian_prod(nodes, nodes)
    
    # Filter out self-loops
    edges = pairs[pairs[:, 0] != pairs[:, 1]]
    
    return edges.T

class HeteroGNNIDS(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, device='cuda', l_features=35,
                 n_features=15, preprocessors=[],
                 **kwargs):
        self.device=device
        
        BaseOnlineODModel.__init__(self,
            model_name='HeteroGNNIDS', n_features=n_features, l_features=l_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
    
        self.G=HeteroData()
        
        #initialize node type features 
        self.G['device'].x=torch.empty((0,n_features)).to(self.device)
        self.G['device'].anomaly_scores=torch.empty(0).to(self.device)
        self.G['device'].idx=torch.empty(0).long().to(self.device)
        
        self.G['device'].mean=torch.zeros(n_features)
        self.G['device'].var=torch.zeros(n_features)
        self.G['device'].count=0
        
        # initialize edge type features        
        self.protocol_map={"UDP":0,"TCP":1,"ARP":2,"ICMP":3,"Other":4}
        self.protocol_inv={v:k for k,v in self.protocol_map.items()}
        
        for link_type in self.protocol_map.keys():
            self.G["device",link_type,"device"].edge_index=torch.empty((2,0)).to(self.device)
            self.G["device",link_type,"device"].edge_attr=torch.empty((0,l_features)).to(self.device)
        
            self.G["device",link_type,"device"].mean=torch.zeros(l_features)
            self.G["device",link_type,"device"].var=torch.zeros(l_features)
            self.G["device",link_type,"device"].count=0
        
    
        self.processed=0
        
        node_types=["device"]
        edge_types=[("device", p, "device") for p in self.protocol_map.keys()]
        
        encoder=GAT(in_channels=n_features, hidden_channels=8, out_channels=2, num_layers=3, v2=True, edge_dim=self.l_features)
        decoder=GAT(in_channels=2, hidden_channels=8, out_channels=n_features, num_layers=3, v2=True, edge_dim=self.l_features)
        self.net=GAE(encoder=to_hetero(encoder,(node_types, edge_types)), decoder=to_hetero(decoder,(node_types, edge_types))).to(self.device)
        
        self.loss=torch.nn.MSELoss(reduction='none').to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-3)
        
        self.additional_params+=["G","processed"]
    
    def to_device(self, x):
        return x.float().to(self.device)
    
    def preprocess_nodes(self, features):
        features=features.cpu()
        
        scaled_features=[]
        # update mean and var of features
        for i in features:
            self.G["device"].count+=1
            delta=i-self.G["device"].mean 
            self.G["device"].mean+=delta/self.G["device"].count 
            delta2=i-self.G["device"].mean
            self.G["device"].var+=delta*delta2

            if self.G["device"].count<2:
                scaled_features.append(torch.zeros(self.n_features))
            else:
                scaled_features.append((features-self.G["device"].mean)/torch.sqrt(self.G["device"].var/self.G["device"].count))
        
        return torch.vstack(scaled_features).to(self.device)
   
    
    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        unique_nodes=torch.unique(torch.cat([srcID, dstID]))
                
        src_feature=self.preprocess_nodes(src_feature).float()
        dst_feature=self.preprocess_nodes(dst_feature).float()
        
        #find set difference unique_nodes-self.G["device"].idx
        uniques, counts = torch.cat((unique_nodes, self.G["device"].idx, self.G["device"].idx)).unique(return_counts=True)
        difference = uniques[counts == 1]

        
        expand_size=len(difference)
            
        if expand_size>0:
            self.G["device"].x=torch.vstack([self.G["device"].x, torch.zeros((expand_size, self.n_features)).to(self.device)])
            self.G["device"].anomaly_scores=torch.hstack([self.G["device"].anomaly_scores, torch.zeros(expand_size).to(self.device)])
            self.G["device"].idx=torch.hstack([self.G["device"].idx, difference.to(self.device)]).long()
            
        #update 
        for i in range(len(srcID)):
            self.G["device"].x[self.G["device"].idx==srcID[i]]=src_feature[i]
            self.G["device"].x[self.G["device"].idx==dstID[i]]=dst_feature[i]
            
        return unique_nodes
    
    
    def preprocess_links(self, features, protocols):
        features=features.cpu()
        
        scaled_features=[]
        
        # update mean and var of features
        for f,p in zip(features, protocols):
            protocol_str=self.protocol_inv[p]
            self.G["device", protocol_str, "device"].count+=1
            delta=f-self.G["device", protocol_str, "device"].mean 
            self.G["device", protocol_str, "device"].mean+=delta/self.G["device", protocol_str, "device"].count 
            delta2=f-self.G["device", protocol_str, "device"].mean
            self.G["device", protocol_str, "device"].var+=delta*delta2
            
            if self.G["device", protocol_str, "device"].count<2:
                scaled_features.append(torch.zeros(self.l_features))
            else:
                scaled_feature=(f-self.G["device", protocol_str, "device"].mean)/torch.sqrt(self.G["device", protocol_str, "device"].var/self.G["device", protocol_str, "device"].count)
                scaled_features.append(torch.nan_to_num(scaled_feature))

        return torch.vstack(scaled_features).float().to(self.device)
        
    def update_edges(self, srcID, dstID, protocol, edge_feature):
        #convert ID to index 
        src_idx = torch.nonzero(self.G["device"].idx == srcID[:, None], as_tuple=False)[:, 1]
        dst_idx = torch.nonzero(self.G["device"].idx == dstID[:, None], as_tuple=False)[:, 1]
        
        protocol=protocol.cpu().numpy()
        edge_feature=self.preprocess_links(edge_feature, protocol)
        
        for p in np.unique(protocol):
            protocol_mask=protocol== p
            proto_str=self.protocol_inv[p]
            
            edge_indices=find_edge_indices(self.G["device",proto_str, "device"].edge_index, src_idx[protocol_mask], dst_idx[protocol_mask])
            existing_edge_idx= edge_indices != -1 
            
            #update found edges if there are any
            if existing_edge_idx.any():
                self.G["device",proto_str, "device"].edge_attr[edge_indices[existing_edge_idx]]=edge_feature[protocol_mask][existing_edge_idx]
            
            num_new_edges=torch.count_nonzero(~existing_edge_idx)
            if num_new_edges>0:
                new_edges=torch.vstack([src_idx[protocol_mask][~existing_edge_idx],dst_idx[protocol_mask][~existing_edge_idx]])
                self.G["device",proto_str, "device"].edge_index=torch.hstack([self.G["device",proto_str, "device"].edge_index, new_edges]).long()
                self.G["device",proto_str, "device"].edge_attr=torch.vstack([self.G["device",proto_str, "device"].edge_attr, edge_feature[protocol_mask][~existing_edge_idx]])   
        
        
    def visualize_graph(self, dataset_name, fe_name, file_name):
        G = to_networkx(self.G, node_attrs=["x","anomaly_scores","idx"], edge_attrs=["edge_attr"],
                        graph_attrs=["threshold"],to_undirected=False, to_multi=True)
        
        
        #indices to label map 
        idx_node_map={i:n for i, n in enumerate(self.G["device"].idx.long().cpu().numpy())}
        node_idx_map={v:k for k,v in idx_node_map.items()}

        node_map=get_node_map(fe_name, dataset_name)
        node_map={v:k for k,v in node_map.items()}

        subset=self.subset.cpu().numpy()
        subset=[node_idx_map[i] for i in subset]
        
        scores=np.array(list(nx.get_node_attributes(G, "anomaly_scores").values()))
        scores=scores-G.graph["threshold"]
        
        node_sm = ScalarMappable(norm=Normalize(vmin=np.min(scores), vmax=max(np.max(scores), G.graph["threshold"])), cmap=plt.cm.coolwarm)
        edge_sm=ScalarMappable(norm=Normalize(0, 5), cmap=plt.cm.Set2)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        non_subset=[]
        node_color=[]
        non_sub_node_color=[]
        for i in list(G):
            if i not in subset:
                non_subset.append(i)
                non_sub_node_color.append(node_sm.to_rgba(scores[i]))
            else:
                node_color.append(node_sm.to_rgba(scores[i]))

        # Visualize the graph using NetworkX
        pos = nx.shell_layout(G)  # Layout algorithm for positioning nodes 
        ax.set_title(f'Network after {self.processed} packets')

        #draw subgraph nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, nodelist=subset, node_shape="s",
                node_color=node_color,
                ax=ax)
        
        #draw all other nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, nodelist=non_subset,  
                node_color=non_sub_node_color, alpha=0.5,
                ax=ax)
        
        
        rad_dict=defaultdict(int)
        for edge in G.edges(data="type"):
            rad=rad_dict[(edge[0],edge[1])]
            edge_color=edge_sm.to_rgba(self.protocol_map[edge[2][1]])
            nx.draw_networkx_edges(G, pos, node_size=800, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {rad}',
                                   edge_color=edge_color, ax=ax)
            rad_dict[(edge[0],edge[1])]+=0.2
            
        # edge_path=nx.draw_networkx_edges(G, pos,  arrowstyle='->',  arrowsize=10, edge_color='gray', ax=ax)
        
        nx.draw_networkx_labels(G, pos, {i: node_map[n] for i, n in nx.get_node_attributes(G,"idx").items()}, font_size=10, ax=ax)
        handles=[]
        for k,v in self.protocol_map.items():
            handles.append(mpatches.Patch(color=edge_sm.to_rgba(v), label=k))
        plt.legend(handles=handles)
        fig.colorbar(node_sm, ax=ax)
        
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
    
         
    def process(self, x):

        self.G.threshold=self.get_threshold()
        
        x=self.preprocess(x)
        x=x.to(self.device)
        
        diff = x[1:, 0] - x[:-1, 0]
        
        split_indices = torch.nonzero(diff, as_tuple=True)[0] + 1
        split_indices=split_indices.tolist()
        split_indices=[0]+split_indices+[x.size(0)]
        split_indices=np.array(split_indices[1:])-np.array(split_indices[:-1])
        updated=False        
        
        for data in torch.split(x, split_indices.tolist(), dim=0):
            #update chunk
            if data[0,0]==1:
                srcIDs, dstIDs, protocols, src_features, dst_features,edge_features=self.split_data(data)

                #find index of last update
                _, indices=uniqueXT(data[:,1:3], return_index=True, occur_last=True, dim=0)
                
                self.subset=self.update_nodes(srcIDs[indices], src_features[indices], dstIDs[indices], dst_features[indices])
                self.update_edges(srcIDs[indices], dstIDs[indices], protocols[indices], edge_features[indices])
                self.processed+=data.size(0)
                updated=True
            # delete links
            else:
                srcIDs, dstIDs, protocols, _, _, _=self.split_data(data)
                protocols=protocols.cpu().numpy()        
                # find edge indices by protocol 
                for p in np.unique(protocols):
                    protocol_mask=protocols== p
                    
                    proto_str=self.protocol_inv[p]
                    
                    edge_indices=find_edge_indices(self.G["device",proto_str, "device"].edge_index, srcIDs[protocol_mask], dstIDs[protocol_mask])
                
                    edge_mask = torch.ones(self.G["device",proto_str, "device"].edge_index.size(1), dtype=torch.bool)
                    edge_mask[edge_indices] = False
                    self.G["device",proto_str, "device"].edge_index=self.G["device",proto_str, "device"].edge_index[:,edge_mask]
                    self.G["device",proto_str, "device"].edge_attr=self.G["device",proto_str, "device"].edge_attr[edge_mask]

                # remove isolated nodes
                
                edges=[]
                splits=[]
                #concat edges 
                for k, v in self.G.edge_index_dict.items():
                    edges.append(v)
                    splits.append(v.size(1))            
                edge_index, _, mask = remove_isolated_nodes(torch.hstack(edges),
                                                                num_nodes=self.G["device"].x.size(0))
                
                #update edge index 
                splited_edges=torch.split(edge_index, splits, dim=1)
                for i, (k, v) in enumerate(self.G.edge_index_dict.items()):
                    self.G[k].edge_index=splited_edges[i]
               
                self.G["device"].x=self.G["device"].x[mask]
                self.G["device"].anomaly_scores=self.G["device"].anomaly_scores[mask]
                self.G["device"].idx=self.G["device"].idx[mask]
             
                # print(self.G.edge_index_dict, self.G.edge_attr_dict)
                
        
        # if only deletion, no need to return anything    
        if not updated:
            return None, None

        self.optimizer.zero_grad()

        # print(self.G["device"].x.shape)

        latent=self.net.encode(self.G.x_dict, self.G.edge_index_dict, edge_attr=self.G.edge_attr_dict)
        recon=self.net.decode(latent, self.G.edge_index_dict, edge_attr=self.G.edge_attr_dict)

        
        loss=self.loss(self.G["device"].x, recon["device"]).mean(1)
        anomaly_scores=torch.log(loss.clone().detach())
        self.G['device'].anomaly_scores=anomaly_scores
        
        anomaly_scores=anomaly_scores.cpu().numpy()
        
        loss=loss.mean()
        loss.backward()
        self.optimizer.step()
            
        self.score_hist.extend(anomaly_scores)
       
        return anomaly_scores, self.G.threshold
            
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.additional_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
                
class Classifier(torch.nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        if channel_list is None:
            self.nn=torch.nn.Identity()
        else:
            self.nn=MLP(channel_list=channel_list)
        self.act=torch.nn.Tanh()
        
    def forward(self, x):
        # Convert node embeddings to edge-level representations:
        x=self.nn(x)
        
        adj_matrix=x@x.T
        
        adj_matrix-=adj_matrix.mean(dim=1)

        # Apply dot-product to get a prediction per supervision edge:
        return self.act(adj_matrix)

class HomoGNNIDS(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, device='cuda', l_features=35,
                 n_features=15, preprocessors=[],
                 **kwargs):
        self.device=device
        
        BaseOnlineODModel.__init__(self,
            model_name='HomoGNNIDS', n_features=n_features, l_features=l_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
    
        self.uniform_sphere_dist = scipy.stats.uniform_direction(2)
    
        self.G=Data()
        
        self.node_hist=deque(maxlen=1000)
        self.edge_hist=deque(maxlen=1000)
        self.struct_hist=deque(maxlen=1000)
        
        #initialize node features 
        self.G.x=torch.empty((0,n_features)).to(self.device)
        self.G.node_as=torch.empty(0).to(self.device)
        self.G.idx=torch.empty(0).long().to(self.device)
        
        self.node_stats=LivePercentile(ndim=n_features)

        #initialize edge features
        self.G.edge_index=torch.empty((2,0)).to(self.device)
        self.G.edge_attr=torch.empty((0,l_features)).to(self.device)
        self.G.edge_as=torch.empty(0).to(self.device)
    
        self.edge_stats=LivePercentile(ndim=l_features)
        
        self.processed=0
        
        self.mac_device_map={"d2:19:d4:e9:94:86":"Router",
                             "22:c9:ca:f6:da:60":"Smartphone 1",
                            "a2:bd:fa:b5:89:92":"Smart Clock 1",
                            "5e:ea:b2:63:fc:aa":"Google Nest Mini 1",
                            "7e:d1:9d:c4:d1:73":"Smart TV",
                            "52:a8:55:3e:34:46":"Smart Bulb 1",
                            "52:8e:44:e1:da:9a":"IP Camera 1",
                            "0a:08:59:8a:2f:1b":"Smart Plug 1",
                            "ea:d8:43:b8:6e:9a":"Raspberry Pi",
                            "ff:ff:ff:ff:ff:ff":"Broadcast",
                            "00:00:00:00:00:00":"Localhost"}
        
        self.node_encoder=GAT(in_channels=n_features, hidden_channels=8, out_channels=2, num_layers=2, v2=True, norm=None, edge_dim=self.l_features).to(self.device)
        self.edge_predictor=Classifier(channel_list=None).to(self.device)
        
        self.edge_decoder=MLP(channel_list=[4,15,35], norm=None).to(self.device)
        self.node_decoder=GAT(in_channels=2, hidden_channels=8, out_channels=n_features, num_layers=2, v2=True, norm=None, edge_dim=self.l_features).to(self.device)
        
        self.prev_node_as=None
        
        self.node_loss=torch.nn.L1Loss(reduction ='none').to(self.device)
        self.edge_loss=torch.nn.L1Loss(reduction='none').to(self.device)
        self.struct_loss=IoULoss().to(self.device)
        
        self.optimizer=torch.optim.Adagrad(self.parameters(),lr=0.001)
        
        
        self.additional_params+=["G","processed","node_hist","node_stats","edge_stats","edge_hist","struct_hist"]
    
    def to_device(self, x):
        return x.float().to(self.device)
    
    def update_nodes_stats(self, features):
        features=features.cpu()
        for i in features:
            self.node_stats.add(i)
    
    def preprocess_nodes(self, features):
        features=features.cpu()
        
        percentiles=self.node_stats.quantiles([0.25,0.50,0.75])
        if percentiles is None:

            percentiles=np.percentile(features.numpy(), [25,50,75], axis=0)
        
        scaled_features= (features-percentiles[1])/(percentiles[2]-percentiles[0])
        return scaled_features.to(self.device).float()
    
    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        unique_nodes=torch.unique(torch.cat([srcID, dstID]))
                
        normalized_src_feature=self.preprocess_nodes(src_feature)
        normalized_dst_feature=self.preprocess_nodes(dst_feature)
        # src_feature=src_feature.float()
        # dst_feature=dst_feature.float()
        
        #find set difference unique_nodes-self.G.node_idx
        uniques, counts = torch.cat((unique_nodes, self.G.idx, self.G.idx)).unique(return_counts=True)
        difference = uniques[counts == 1]

        expand_size=len(difference)
            
        if expand_size>0:
            self.G.x=torch.vstack([self.G.x, torch.zeros((expand_size, self.n_features)).to(self.device)])
            self.G.node_as=torch.hstack([self.G.node_as, torch.zeros(expand_size).to(self.device)])
            self.G.idx=torch.hstack([self.G.idx, difference.to(self.device)]).long()
            
        #update 
        for i in range(len(srcID)):
            self.G.x[self.G.idx==srcID[i]]=normalized_src_feature[i]
            self.G.x[self.G.idx==dstID[i]]=normalized_dst_feature[i]
        
        # update feature
        self.update_nodes_stats(src_feature)
        self.update_nodes_stats(dst_feature) 
        
        return unique_nodes
    
    def update_link_stats(self, features):
        features=features.cpu()
        
        for f in features:
            self.edge_stats.add(f)
            
    def preprocess_links(self, features):
        features=features.cpu()
        
        percentiles=self.edge_stats.quantiles([0.25,0.50,0.75])
        if percentiles is None:
            percentiles=np.percentile(features, [25,50,75], axis=0)
            
        scaled_features= (features-percentiles[1])/(percentiles[2]-percentiles[0])
        return scaled_features.to(self.device).float()        

    def update_edges(self, srcID, dstID, protocol, edge_feature):
        #convert ID to index 
        src_idx = torch.nonzero(self.G.idx == srcID[:, None], as_tuple=False)[:, 1]
        dst_idx = torch.nonzero(self.G.idx == dstID[:, None], as_tuple=False)[:, 1]
        
        normalized_edge_feature=self.preprocess_links(edge_feature)
        # edge_feature=edge_feature.float()
    
        edge_indices=find_edge_indices(self.G.edge_index, src_idx, dst_idx)
        existing_edge_idx= edge_indices != -1 
        
        #update found edges if there are any
        if existing_edge_idx.any():
            self.G.edge_attr[edge_indices[existing_edge_idx]]=normalized_edge_feature[existing_edge_idx]
        
        num_new_edges=torch.count_nonzero(~existing_edge_idx)
        if num_new_edges>0:
            new_edges=torch.vstack([src_idx[~existing_edge_idx],dst_idx[~existing_edge_idx]])
            self.G.edge_index=torch.hstack([self.G.edge_index, new_edges]).long()
            self.G.edge_attr=torch.vstack([self.G.edge_attr, normalized_edge_feature[~existing_edge_idx]])
            self.G.edge_as=torch.hstack([self.G.edge_as, torch.zeros(num_new_edges).to(self.device)])  
        
        self.update_link_stats(edge_feature)
        
    def visualize_graph(self, dataset_name, fe_name, file_name):
        G = to_networkx(self.G, node_attrs=["x","node_as","idx"], edge_attrs=["edge_attr","edge_as"],
                        graph_attrs=["threshold","struct_as","node_thresh","edge_thresh","struct_thresh"], to_undirected=False, to_multi=True)
        
        
        #indices to label map 
        idx_node_map={i:n for i, n in enumerate(self.G.idx.long().cpu().numpy())}
        node_idx_map={v:k for k,v in idx_node_map.items()}

        node_map=get_node_map(fe_name, dataset_name)
        node_map={v:k for k,v in node_map.items()}

        subset=self.subset.cpu().numpy()
        subset=[node_idx_map[i] for i in subset]
        
        node_as=np.array(list(nx.get_node_attributes(G, "node_as").values()))
        node_as-=G.graph["node_thresh"]
        
        edge_as=np.array(list(nx.get_edge_attributes(G, "edge_as").values()))
        edge_as-=G.graph["edge_thresh"]
        
        
        node_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=min(np.min(node_as),-1), vcenter=0., vmax=max(np.max(node_as), 1)), cmap=plt.cm.coolwarm)
        edge_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=min(np.min(edge_as),-1), vcenter=0., vmax=max(np.max(edge_as), 1)), cmap=plt.cm.coolwarm)
        
        fig, ax = plt.subplots(figsize=(8, 5))

        

        # Visualize the graph using NetworkX
        pos = nx.shell_layout(G)  # Layout algorithm for positioning nodes 
        ax.set_title(f'Network after {self.processed} packets Struct Loss {G.graph["struct_as"]-G.graph["struct_thresh"]:.2f}')

        #draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800,
                node_color=[node_sm.to_rgba(i) for i in node_as],
                ax=ax)
        
        nx.draw_networkx_edges(G, pos, node_size=800, arrowstyle='->',  arrowsize=10, edge_color=[edge_sm.to_rgba(i) for i in edge_as], ax=ax)
        
        #draw labels for subset
        nx.draw_networkx_labels(G, pos, {i: self.mac_device_map.get(node_map[n], node_map[n]) for i, n in nx.get_node_attributes(G,"idx").items()
                                         if i in subset}, font_size=10,font_weight="bold", ax=ax)
        
        #draw labels for non subset
        nx.draw_networkx_labels(G, pos, {i: self.mac_device_map.get(node_map[n], node_map[n]) for i, n in nx.get_node_attributes(G,"idx").items()
                                         if i not in subset}, font_size=10, ax=ax)
        
        node_cbar=fig.colorbar(node_sm, ax=ax, location="right")
        node_cbar.ax.set_ylabel('node anomaly score', rotation=90)
        edge_cbar=fig.colorbar(edge_sm, ax=ax, location="left")
        edge_cbar.ax.set_ylabel('edge anomaly score', rotation=90)

        fig.tight_layout()
        path=Path(f"../../datasets/{dataset_name}/{fe_name}/graph_plots/{file_name}/{self.model_name}_{self.processed}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format='png')
        # Close the figure to release resources
        plt.close(fig)
    
    def split_data(self, x):
        srcIDs=x[:, 1].long()
        dstIDs=x[:, 2].long()
        protocols=x[:, 3].long()
        
        src_features=x[:, 4:19].float()
        dst_features=x[:, 19:34].float()
        edge_features=x[:, 34:].float()
        
        return srcIDs, dstIDs, protocols, src_features, dst_features,edge_features
    
    def sample_edges(self, p):
        n=self.G.x.size(0)
        probabilities = torch.rand(n, n)
        
        # Get indices where probabilities are less than p and apply the mask
        indices = torch.argwhere(probabilities < p).T 
        
        return indices, edge_exists(self.edge_idx, indices)
    
    
    def get_threshold(self):
        all_thresh= super().get_threshold()
        if all_thresh == 0:
            node_thresh=0
            edge_thresh=0
            struct_thresh=0
        else:
            node_thresh=np.percentile(self.node_hist, 99.9)
            edge_thresh=np.percentile(self.edge_hist, 99.9)
            struct_thresh=np.percentile(self.struct_hist, 99.9)
        return all_thresh, node_thresh, edge_thresh, struct_thresh
    
    def decode_graph(self, latent_node):
        adj_matrix=self.edge_predictor(latent_node)
        #convert to predicted edge_indices 
        pred_edge_index=(adj_matrix>0.5).nonzero().T
        
        #decode edges 
        latent_edge_tuple=torch.hstack([latent_node[self.G.edge_index[0]], latent_node[self.G.edge_index[1]]])
        decoded_edge_attr=self.edge_decoder(latent_edge_tuple)
        
        #decode node
        pred_edge_tuple=torch.hstack([latent_node[pred_edge_index[0]],latent_node[pred_edge_index[1]]])
        pred_edge_attr=self.edge_decoder(pred_edge_tuple)
        decoded_node=self.node_decoder(latent_node, pred_edge_index, pred_edge_attr)
        return pred_edge_index, decoded_edge_attr, decoded_node
    
    def process(self, x):
        self.G.threshold, self.G.node_thresh, self.G.edge_thresh, self.G.struct_thresh=self.get_threshold()
        
        x=self.preprocess(x)
        x=x.to(self.device)
        
        diff = x[1:, 0] - x[:-1, 0]
        
        split_indices = torch.nonzero(diff, as_tuple=True)[0] + 1
        split_indices=split_indices.tolist()
        split_indices=[0]+split_indices+[x.size(0)]
        split_indices=np.array(split_indices[1:])-np.array(split_indices[:-1])
        updated=False
        
        for data in torch.split(x, split_indices.tolist(), dim=0):
            #update chunk
            if data[0,0]==1:
                srcIDs, dstIDs, protocols, src_features, dst_features,edge_features=self.split_data(data)

                #find index of last update
                _, indices=uniqueXT(data[:,1:3], return_index=True, occur_last=True, dim=0)
                self.subset=self.update_nodes(srcIDs[indices], src_features[indices], dstIDs[indices], dst_features[indices])
                self.update_edges(srcIDs[indices], dstIDs[indices], protocols[indices], edge_features[indices])
                updated=True
                self.processed+=data.size(0)
                
            # delete links
            else:
                srcIDs, dstIDs, protocols, _, _, _=self.split_data(data)

                # convert to idx
                src_idx = torch.nonzero(self.G.idx == srcIDs[:, None], as_tuple=False)[:, 1]
                dst_idx = torch.nonzero(self.G.idx == dstIDs[:, None], as_tuple=False)[:, 1]
                
                edge_indices=find_edge_indices(self.G.edge_index, src_idx, dst_idx)
                
                edge_mask = torch.ones(self.G.edge_index.size(1), dtype=torch.bool)
                edge_mask[edge_indices] = False
                
                self.G.edge_index=self.G.edge_index[:,edge_mask]
                self.G.edge_attr=self.G.edge_attr[edge_mask]
                self.G.edge_as=self.G.edge_as[edge_mask]

                # remove isolated nodes
                edge_index, _, mask = remove_isolated_nodes(self.G.edge_index,
                                                                num_nodes=self.G.x.size(0))
                
                #update edge index 
                self.G.edge_index=edge_index
                self.G.x=self.G.x[mask]
                self.G.node_as=self.G.node_as[mask]
                self.G.idx=self.G.idx[mask]
                self.processed+=data.size(0)
                
 
        # if only deletion, no need to return anything    
        if not updated:
            return None, None

        self.optimizer.zero_grad()
        
        latent_node=self.node_encoder(self.G.x, self.G.edge_index, edge_attr=self.G.edge_attr)
        pred_edge_index, decoded_edge_attr, decoded_node=self.decode_graph(latent_node)
        
        # scores
        node_loss=self.node_loss(decoded_node,self.G.x)
        edge_loss=self.edge_loss(decoded_edge_attr, self.G.edge_attr)
        structural_loss=self.struct_loss(pred_edge_index, self.G.edge_index, self.G.x.size(0))
        
        self.G.node_as=node_loss.mean(1).clone().detach()
        self.G.edge_as=edge_loss.mean(1).clone().detach()
        self.G.struct_as=structural_loss.clone().detach()
        
        loss=node_loss.mean()+edge_loss.mean()+structural_loss #-contrastive_loss.mean()
        loss.backward()
        self.optimizer.step()
        
        loss_np=loss.clone().detach().cpu().numpy()
        self.score_hist.append(loss_np)
        self.node_hist.extend(self.G.node_as.cpu().numpy())
        self.edge_hist.extend(self.G.edge_as.cpu().numpy())
        self.struct_hist.append(self.G.struct_as.cpu().numpy())
        
        return {"threshold":self.G.threshold, "node_loss":node_loss.detach().cpu().numpy(),
                "edge_loss":edge_loss.detach().cpu().numpy(), "structural_loss":structural_loss.detach().cpu().numpy(),
                "desc":f" nl {node_loss.mean():.3f}, el {edge_loss.mean():.3f}, sl {structural_loss:.3f}"}
    
    
    def calc_sw_loss(self, latent):
        theta = generate_theta(
            50, latent.size(1)).float().to(self.device)

        # Define a Keras Variable for samples of z
        z = generate_z(latent.size(0), latent.size(1),
                       "circular", center=0).float().to(self.device)

        # Let projae be the projection of the encoded samples
        projae = torch.tensordot(latent, theta.T, dims=1)
        # projae += tf.expand_dims(tf.norm(encoded, axis=1), axis=1)
        # Let projz be the projection of the $q_Z$ samples
        projz = torch.tensordot(z, theta.T, dims=1)
        # projz += tf.expand_dims(tf.norm(z, axis=1), axis=1)
        # Calculate the Sliced Wasserstein distance by sorting
        # the projections and calculating the L2 distance between
        sw_loss = ((torch.sort(projae.T).values
                               - torch.sort(projz.T).values)**2).mean()

        return sw_loss
    
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            # save tdigests as centroids
            if i in ["node_stats","edge_stats"]:
                digests=getattr(self, i)
                state[i]=digests.to_centroids()
            else:    
                state[i] = getattr(self, i)
            
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.additional_params:
            if i in ["node_stats","edge_stats"]:
                dim_list=state_dict[i]
                live_stats=LivePercentile(dim_list)
                setattr(self, i, live_stats)
            else:
                setattr(self, i, state_dict[i])
            del state_dict[i]
                