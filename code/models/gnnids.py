import torch
from math import copysign,fabs,sqrt
import numpy as np
from .base_model import BaseOnlineODModel, TorchSaveMixin,EnsembleSaveMixin
from torch_geometric.data import Data, HeteroData
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.models import GAT,GAE,MLP, GCN
from torch_geometric.utils import k_hop_subgraph,to_networkx, remove_isolated_nodes
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import pickle 
from utils import *
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,TwoSlopeNorm,LogNorm
import matplotlib.patches as mpatches
import torch_geometric.transforms as T
from collections import deque
import scipy
from scipy import integrate
from pytdigest import TDigest
from tqdm import tqdm
torch.set_printoptions(precision=4)

def calc_quantile(x, p):
    x=np.log(x)
    mean=np.mean(x) 
    std=np.std(x)
    # print(mean, std)
    quantile=np.exp(mean+np.sqrt(2)*std*scipy.special.erfinv(2*p-1))
    return quantile

def is_stable(x, p=0.9, return_quantile=False):
    if len(x)!=x.maxlen:
        stability=False
        quantile=0.
    else:
        quantile=calc_quantile(x, p)
        stability=(np.array(x)<quantile).all()
        
    if return_quantile:
        return stability, quantile
    else:
        return stability

class SWLoss(torch.nn.Module):
    def __init__(self,slices, shape):
        super().__init__()
        self.slices=slices
        self.shape=shape
        
    def forward(self, encoded):
        # Define a PyTorch Variable for theta_ls
        theta = generate_theta(self.slices, encoded.size(1)).to("cuda")

        # Define a PyTorch Variable for samples of z
        z = generate_z(encoded.size(0), encoded.size(1),
                        self.shape, center=0).to("cuda")

        # Let projae be the projection of the encoded samples
        projae = torch.tensordot(encoded, theta.T, dims=1)


        # Let projz be the projection of the q_Z samples
        projz = torch.tensordot(z, theta.T, dims=1)

        # Calculate the Sliced Wasserstein distance by sorting
        # the projections and calculating the L2 distance between
        sw_loss = ((torch.sort(projae.T)[0]
                                - torch.sort(projz.T)[0])**2).mean()
        return sw_loss

class AE(torch.nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, in_channels),
        )
 
    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LivePercentile:
    def __init__(self, ndim=None):
        """ Constructs a LiveStream object
        """
        
        if isinstance(ndim, int):
            self.dims=[TDigest() for _ in range(ndim)]
            self.patience=0
            self.ndim=ndim
        elif isinstance(ndim, list):
            self.dims=self.of_centroids(ndim)
            self.ndim=len(ndim)
            self.patience=10
        else:
            raise ValueError("ndim must be int or list")
        

    def add(self, item):
        """ Adds another datum """
        if isinstance(item, torch.Tensor):
            item=item.numpy()
        
        if self.ndim==1:
            self.dims[0].update(item)
        else:
            for i, n in enumerate(item.T):
                self.dims[i].update(n)
        
        self.patience+=1

    def reset(self):
        self.dims=[TDigest() for _ in range(self.ndim)]
        self.patience=0
    
    def quantiles(self, p):
        """ Returns a list of tuples of the quantile and its location """
        if self.patience<1:
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
    return torch.tensor(rand/theta).float()


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
    elif shape=="gaussian":
        z=np.random.normal(size=[n,dim])
    elif shape=="log-normal":
        z=np.exp(np.random.normal(size=[n,dim]))
    return torch.tensor(z).float()


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
    try:
        with open(f"../../datasets/{dataset_name}/{fe_name}/state.pkl", "rb") as pf:
            state=pickle.load(pf)
            
        return state["node_map"]
    except FileNotFoundError as e:
        return None
    

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
            self.nn=MLP(channel_list=channel_list,norm=None) #
        self.act=torch.nn.Sigmoid()
        
    def forward(self, x):
        # Convert node embeddings to edge-level representations:
        x=self.nn(x)
        adj_matrix=x@x.T

        # adj_matrix-=adj_matrix.min(dim=0, keepdims=True).values
        # Apply dot-product to get a prediction per supervision edge:
        return self.act(adj_matrix)

class VariationalGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.gcn_shared(x, edge_index, edge_attr))
        mu = self.gcn_mu(x, edge_index, edge_attr)
        logvar = self.gcn_logvar(x, edge_index, edge_attr)
        return mu, logvar

class PermutationEquivariantEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.node_embed = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=2, norm=None)
        self.linear=torch.nn.Linear(out_channels,out_channels)
        
        
    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings=self.node_embed(x,edge_index, edge_attr)
        return self.linear(node_embeddings)


class Diffusion:
    def __init__(self, noise_steps=20, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:,None]
        Ɛ = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x

class ReluBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.dense=torch.nn.Linear(time_dim, out_channels)
        
    def forward(self, x, t=None):
        x = self.linear(x)
        x = self.relu(x)
        
        if t is not None:
            x=x+self.dense(t)
                
        return x
    
class NoisePredictor(torch.nn.Module):
    def __init__(self, in_channels, time_dim=16):
        super().__init__()
        self.enc1=ReluBlock(in_channels, 16)
        self.enc2=ReluBlock(16, 8)
        self.enc3=ReluBlock(8, 4)
        
        self.dec2=ReluBlock(4, 8)
        self.dec3=ReluBlock(8, 16)
        self.dec4=ReluBlock(16, in_channels)
        
        self.device="cuda"
        
        self.out=torch.nn.Linear(in_channels,in_channels)
        
        self.pos_encoding=PositionalEncoding(20, time_dim)
        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                time_dim,
                in_channels
            ),
        )
        
    def forward(self, x, t):
        t = self.pos_encoding(t)
        x=x+self.emb_layer(t)
        
        x1=self.enc1(x)
        x2=self.enc2(x1)
        x3=self.enc3(x2)
        
        x4=self.dec2(x3)+x2
        x5=self.dec3(x4)+x1
        x6=self.dec4(x5)
        
        return self.out(x6)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_time_steps, embedding_size, n=10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to("cuda")
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]

class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False).to("cuda")
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNet(torch.nn.Module):
    def __init__(self, marginal_prob_std, in_channels, time_dim=16):
        super().__init__()
        self.enc1=ReluBlock(in_channels, 16, time_dim)
        self.enc2=ReluBlock(16, 8, time_dim)
        self.enc3=ReluBlock(8, 4, time_dim)
        
        self.dec2=ReluBlock(4, 8, time_dim)
        self.dec3=ReluBlock(8, 16, time_dim)
        self.dec4=ReluBlock(16, in_channels, time_dim)
        
        self.device="cuda"
        # The swish activation function
        self.act = torch.nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        self.out=torch.nn.Linear(in_channels,in_channels)
        
        self.embed = torch.nn.Sequential(GaussianFourierProjection(embed_dim=time_dim),
         torch.nn.Linear(time_dim, time_dim))
        
    def forward(self, x, t):
        embed = self.act(self.embed(t))
        
        x1=self.enc1(x, embed)
        x2=self.enc2(x1, embed)
        x3=self.enc3(x2, embed)
        
        x4=self.dec2(x3, embed)+x2
        x5=self.dec3(x4, embed)+x1
        x6=self.dec4(x5, embed)
        
        return self.out(x6) / self.marginal_prob_std(t)[:,None]

def ode_likelihood(x, 
                   score_model,
                   marginal_prob_std, 
                   diffusion_coeff,
                   batch_size=64, 
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.
    
    Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the 
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the 
        forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)
        
    def divergence_eval(sample, time_steps, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1))    
    
    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
        # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((shape[0],)) * t    
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([x.cpu().detach().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    return z, prior_logp

def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1)) / (2 * sigma**2)

def marginal_prob_std(t):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """
    sigma=torch.tensor(25)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / torch.log(sigma))

def diffusion_coeff(t, ):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    
    Returns:
        The vector of diffusion coefficients.
    """
    sigma=torch.tensor(25)
    return sigma**t

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, random_t)
    loss = torch.sum((score * std[:, None] + z)**2, dim=(1))
    return loss

class HomoGNNIDS(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, device='cuda', l_features=35,
                 n_features=15, preprocessors=[], model_name="HomoGNNIDS",
                 node_latent_dim=5,
                 use_edge_attr=True, 
                 **kwargs):
        self.device=device
        
        BaseOnlineODModel.__init__(self,
            model_name=model_name, n_features=n_features, l_features=l_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
    
        self.uniform_sphere_dist = scipy.stats.uniform_direction(2)
        self.node_latent_dim=node_latent_dim
        
        self.use_edge_attr=use_edge_attr
        
        #initialize graph
        self.G=Data()
        
        #initialize node features 
        self.G.x=torch.empty((0,n_features)).to(self.device)
        self.G.node_as=torch.empty(0).to(self.device)
        self.G.idx=torch.empty(0).long().to(self.device)
        
        #initialize edge features
        self.G.edge_index=torch.empty((2,0)).to(self.device)
        self.G.edge_as=torch.empty(0).to(self.device)
        
        if self.use_edge_attr:
            self.G.edge_attr=torch.empty((0,l_features)).to(self.device)    

        self.n=100
        self.loss_queue=deque(maxlen=self.n)
        self.potential_queue=deque(maxlen=self.n)
        self.potential_x_queue=deque(maxlen=self.n)
        self.potential_con_queue=deque(maxlen=self.n)
        
        self.model_idx=0
    
        self.processed=0
        self.subset=[]
        
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
                            "00:00:00:00:00:00":"Localhost",
                            "be:7b:f6:f2:1b:5f":"Attacker"}
        
        self.mse=torch.nn.MSELoss(reduction ='none').to(self.device)
        self.sw_loss=SWLoss(50, "log-normal").to(self.device)
        
        if self.use_edge_attr:
            self.edge_loss=torch.nn.L1Loss(reduction='none').to(self.device)
            self.additional_params+=["edge_stats"]
        
        self.model_pool=[self.create_model()]

        self.additional_params+=["G", "processed", "loss_queue", "potential_queue", "model_pool", "model_idx"]
        
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def create_model(self):
        model_dict={}
        model_dict["loss_hist"]=deque(maxlen=self.n)
        model_dict["anomaly_scores"]=LivePercentile(ndim=1)
        model_dict["node_stats"]=LivePercentile(ndim=self.n_features)
        if self.use_edge_attr:
            model_dict["edge_stats"]=LivePercentile(ndim=self.l_features)
            model_dict["edge_decoder"]=MLP(channel_list=[2*self.node_latent_dim,35,35], norm=None).to(self.device)
        else:
            model_dict["edge_decoder"]=None
            
        model_dict["node_encoder"]=PermutationEquivariantEncoder(in_channels=self.n_features, hidden_channels=self.n_features, out_channels=self.node_latent_dim).to(self.device)
        # model_dict["diffusion"]=Diffusion(img_size=self.node_latent_dim)
        # model_dict["noise_predictor"]=NoisePredictor(in_channels=self.node_latent_dim).to(self.device)
        
        # model_dict["score_model"]=ScoreNet(marginal_prob_std,16).to(self.device)
        model_dict["outlier_detector"]=AE(in_channels=self.node_latent_dim).to(self.device)
        
        model_dict["optimizer"]=torch.optim.Adam(list(model_dict["node_encoder"].parameters())+
                                                 list(model_dict["outlier_detector"].parameters()),lr=0.001)
        
        model_dict["converged"]=False
        model_dict["threshold"]=0
        
        return model_dict
    
    def to_device(self, x):
        return x.float().to(self.device)
    
    def preprocess_features(self, features, live_stats):
        features=features.cpu()
        
        percentiles=live_stats.quantiles([0.25,0.50,0.75])
        if percentiles is None:
            percentiles=np.percentile(features.numpy(), [25,50,75], axis=0)
        
        scaled_features= (features-percentiles[1])/(percentiles[2]-percentiles[0])
        scaled_features=torch.nan_to_num(scaled_features, nan=0., posinf=0., neginf=0.)
        return scaled_features.to(self.device).float()
    
    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        unique_nodes=torch.unique(torch.cat([srcID, dstID]))
                
        # normalized_src_feature=self.preprocess_nodes(src_feature)
        # normalized_dst_feature=self.preprocess_nodes(dst_feature)
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
            self.G.x[self.G.idx==srcID[i]]=src_feature[i]
            self.G.x[self.G.idx==dstID[i]]=dst_feature[i]
        
        return unique_nodes.tolist()
            

    def update_edges(self, srcID, dstID, protocol, edge_feature):
        #convert ID to index 
        src_idx = torch.nonzero(self.G.idx == srcID[:, None], as_tuple=False)[:, 1]
        dst_idx = torch.nonzero(self.G.idx == dstID[:, None], as_tuple=False)[:, 1]
        
    
        edge_indices=find_edge_indices(self.G.edge_index, src_idx, dst_idx)
        existing_edge_idx= edge_indices != -1 
        
        if self.use_edge_attr:
            #update found edges if there are any
            if existing_edge_idx.any():
                self.G.edge_attr[edge_indices[existing_edge_idx]]=edge_feature[existing_edge_idx]
        
        num_new_edges=torch.count_nonzero(~existing_edge_idx)
        if num_new_edges>0:
            new_edges=torch.vstack([src_idx[~existing_edge_idx],dst_idx[~existing_edge_idx]])
            self.G.edge_index=torch.hstack([self.G.edge_index, new_edges]).long()
            
            if self.use_edge_attr:
                self.G.edge_attr=torch.vstack([self.G.edge_attr, edge_feature[~existing_edge_idx]])
            self.G.edge_as=torch.hstack([self.G.edge_as, torch.zeros(num_new_edges).to(self.device)])  
        
        if self.use_edge_attr:
            self.edge_stats.add(edge_feature.cpu())
    
    def visualize_graph(self, dataset_name, fe_name, file_name):
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        node_map=get_node_map(fe_name, dataset_name)
        node_map={v:k for k,v in node_map.items()}
        
        self.draw_graph(self, fig, ax, node_map)
        
        fig.tight_layout()
        path=Path(f"../../datasets/{dataset_name}/{fe_name}/graph_plots/{file_name}/{self.model_name}_{self.processed}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format='png')
        # Close the figure to release resources
        plt.close(fig)
        
    def draw_graph(self, fig, ax, node_map):
        
        G = to_networkx(self.G, node_attrs=["node_as","idx"], edge_attrs=["edge_as"],
                     to_undirected=False, to_multi=True)
        
        #indices to label map 
        idx_node_map={i:n for i, n in enumerate(self.G.idx.long().cpu().numpy())}
        node_idx_map={v:k for k,v in idx_node_map.items()}

        
        subset=[node_idx_map[i] for i in self.subset]
        
        node_as=np.array(list(nx.get_node_attributes(G, "node_as").values()))
        # node_as-=G.graph["node_thresh"]
        
        edge_as=np.array(list(nx.get_edge_attributes(G, "edge_as").values()))
        # edge_as-=G.graph["edge_thresh"]
        
        
        # node_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=min(np.min(node_as),-1), vcenter=0., vmax=max(np.max(node_as), 1)), cmap=plt.cm.coolwarm)
        # edge_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=min(np.min(edge_as),-1), vcenter=0., vmax=max(np.max(edge_as), 1)), cmap=plt.cm.coolwarm)
        
        node_sm = ScalarMappable(norm=Normalize(vmin=np.min(node_as),  vmax=np.max(node_as)), cmap=plt.cm.winter)
        edge_sm = ScalarMappable(norm=Normalize(vmin=np.min(edge_as),  vmax=np.max(edge_as)), cmap=plt.cm.winter)
        

        # Visualize the graph using NetworkX
        pos = nx.shell_layout(G)  # Layout algorithm for positioning nodes 

        #draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800,
                node_color=[node_sm.to_rgba(i) for i in node_as],
                ax=ax)
        
        # nx.draw_networkx_edges(G, pos, node_size=800, arrowstyle='->',  arrowsize=10, edge_color=[edge_sm.to_rgba(i) for i in edge_as], ax=ax)
        
        rad_dict=defaultdict(int)
        for i, edge in enumerate(G.edges()):
            rad=rad_dict[frozenset([edge[0],edge[1]])]
            edge_color=edge_sm.to_rgba(edge_as[i])
            nx.draw_networkx_edges(G, pos, node_size=800, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {rad}',
                                    arrowstyle='->',  arrowsize=20,
                                   edge_color=edge_color, ax=ax)
            rad_dict[frozenset([edge[0],edge[1]])]+=0.2
            
        # identity mapping if no node map
        if node_map is None:
            
            node_map={i:i for i in range(self.G.idx.max()+1)}
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
    
    
    def decode_graph(self, latent_node, node_decoder, edge_decoder):
        
        #decode edges 
        if self.use_edge_attr:
            latent_edge_tuple=torch.hstack([latent_node[self.G.edge_index[0]], latent_node[self.G.edge_index[1]]])
            decoded_edge_attr=edge_decoder(latent_edge_tuple)
        
        #decode node
        else:
            decoded_edge_attr=None
            
        decoded_node=node_decoder(x=latent_node, edge_index=self.G.edge_index, edge_attr=self.G.edge_attr)
        return decoded_edge_attr, decoded_node
            
    def process(self, x):
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
                srcIDs, dstIDs, protocols, src_features, dst_features, edge_features=self.split_data(data)
                
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
                
                if self.use_edge_attr:
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
            return None

        #try with existing index
        node_loss,node_embed = self.eval_model(self.model_pool[self.model_idx])
        
        loss_diff=node_loss.mean()-self.model_pool[self.model_idx]['threshold']
        # if model has not converged or is benign, train it
        if not self.model_pool[self.model_idx]["converged"] or loss_diff<0:
            
            train_loss = self.train_model(self.model_pool[self.model_idx])
                
            # self.model_pool[self.model_idx]["node_stats"].add(self.G.x.cpu())
            # if self.use_edge_attr:
            #     self.model_pool[self.model_idx]["edge_stats"].add(self.G.edge_attr.cpu())
            
            self.model_pool[self.model_idx]["loss_hist"].append(train_loss)
            self.model_pool[self.model_idx]["anomaly_scores"].add(np.log(train_loss))  
            self.model_pool[self.model_idx]["threshold"]=np.exp(self.model_pool[self.model_idx]["anomaly_scores"].quantiles([0.99]).item())
            
            if is_stable(self.model_pool[self.model_idx]["loss_hist"]):
                self.model_pool[self.model_idx]["converged"]=True
        
            # add average AS to queue
            self.loss_queue.append(train_loss)
            #empty potential queue 
            self.potential_queue.clear()
            
            self.potential_x_queue.clear()
            self.potential_con_queue.clear()
        
        
        #if converged and loss_diff is greater than 0
        else:
            for i, model in enumerate(self.model_pool):
                tmp_node_loss, tmp_node_embed=self.eval_model(model)
                
                tmp_loss_diff=tmp_node_loss.mean()-model["threshold"]
                if tmp_loss_diff<loss_diff:
                     
                    loss_diff=tmp_loss_diff
                    node_loss=tmp_node_loss
                    node_embed=tmp_node_embed
                    
                    self.model_idx=i 

            #if still anomaly, add to potential queue
            if loss_diff>0:
                self.potential_queue.append(node_loss.mean().cpu().item())
                self.potential_x_queue.append(self.G.x.clone())
                self.potential_con_queue.append(self.G.edge_index.clone())
            else:
                self.loss_queue.append(node_loss.mean().cpu().item())

        if len(self.loss_queue)==self.loss_queue.maxlen and len(self.potential_queue)==self.potential_queue.maxlen: 
            difference=np.median(self.potential_queue)-np.median(self.loss_queue)
            if difference > calc_quantile(self.loss_queue, 0.95):
                drift_level="malicious"
            else:
                drift_level="benign"
                new_model= self.create_model()
                #train model on existing features 
                for prev_x, prev_edge_idx in zip(self.potential_x_queue, self.potential_con_queue):
                    train_loss=self.train_model(new_model, prev_x, prev_edge_idx)
                    new_model["node_stats"].add(prev_x.cpu())
                    if self.use_edge_attr:
                        new_model["edge_stats"].add(self.G.edge_attr.cpu())
                
                node_loss, node_embed=self.eval_model(new_model)
                    
                self.model_idx=len(self.model_pool)
                self.model_pool.append(new_model)
                
        elif len(self.loss_queue)!=self.loss_queue.maxlen:
            drift_level="unfull loss queue"
        else:
            drift_level="no drift"
        
        self.G.node_as=node_loss
        if self.use_edge_attr:
            self.G.edge_as=edge_loss.clone().detach()
        else:
            self.G.edge_as=torch.zeros(self.G.edge_index.size(1)).to(self.device)
        
        # add to node loss
        node_loss_dict={idx:self.G.node_as[i].item() for i,idx in enumerate(self.G.idx.cpu().numpy())}
        
        results_dict={
            "threshold": self.model_pool[self.model_idx]["threshold"],
            "loss": node_loss.mean().detach().cpu().numpy(),
            "node_loss":node_loss_dict,
            "node_embed": node_embed.float().detach().cpu().numpy(),
            "drift_level": drift_level,
            "num_model":len(self.model_pool),
            "model_index":self.model_idx, 
            "pool_stable":self.model_pool[self.model_idx]["converged"]
        }

        
        return results_dict

    def eval_model(self, model):
        self.eval()
        
        normalized_x=self.preprocess_features(self.G.x, model["node_stats"])
            
        if self.use_edge_attr:
            normalized_edge_attr=self.preprocess_features(self.G.edge_attr, model["edge_stats"])
        else:
            normalized_edge_attr=None

        node_embed=model['node_encoder'](x=normalized_x, edge_index=self.G.edge_index, edge_attr=normalized_edge_attr)
        # z, prior_logp=ode_likelihood(node_embed, model["score_model"], marginal_prob_std, diffusion_coeff,node_embed.shape[0])
        recon=model["outlier_detector"](node_embed)
        
        loss=self.mse(node_embed,recon).mean(dim=1)
        # loss=loss_fn(model["score_model"], node_embed, marginal_prob_std)
        
        return loss.detach(), node_embed
        

    def train_model(self, model, x=None, edge_idx=None):
        self.train()
        
        if x is None:
            normalized_x=self.preprocess_features(self.G.x, model["node_stats"])
            edge_idx=self.G.edge_index
        else:
            normalized_x=self.preprocess_features(x, model["node_stats"])
        
            
        if self.use_edge_attr:
            normalized_edge_attr=self.preprocess_features(self.G.edge_attr, model["edge_stats"])
        else:
            normalized_edge_attr=None
        
        
        model["optimizer"].zero_grad()
        node_embed=model['node_encoder'](x=normalized_x, edge_index=edge_idx, edge_attr=normalized_edge_attr)
        
        recon=model["outlier_detector"](node_embed)
        
        node_loss=self.mse(node_embed,recon).mean()
        
        loss=node_loss+self.sw_loss(node_embed)
        # loss=loss_fn(model["outlier_detector"], node_embed, marginal_prob_std).mean()
        
        loss.backward()
        model["optimizer"].step()
        
        # if self.training:
        #     t=model["diffusion"].sample_timesteps(normalized_x.shape[0]).to(self.device)
        # else:
        #     t=torch.full(size=(normalized_x.size(0),),fill_value=1)
            
        # x_t, noise = model["diffusion"].noise_images(node_embed, t)
        # predicted_noise = model["noise_predictor"](x_t, t)
        
        # node_loss = self.node_loss(noise, predicted_noise).mean(dim=1)
        # if self.training:
        #     node_loss/=torch.linalg.norm(noise, dim=1)
        
        return node_loss.detach().cpu().item()
    
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            if i=="model_pool":
                # save tdigests as centroids
                for model in self.model_pool:
                    model["node_stats"]=model["node_stats"].to_centroids()
                    model["anomaly_scores"]=model["anomaly_scores"].to_centroids()
                    if self.use_edge_attr:
                        model["edge_stats"]=model["edge_stats"].to_centroids()
                    
                        
            state[i] = getattr(self, i)
            
        return state
    
    def load_state_dict(self, state_dict):
        for i in self.additional_params:
            if i=="model_pool":
                for j, model in enumerate(state_dict[i]):
                    state_dict[i][j]["node_stats"]=LivePercentile(state_dict[i][j]["node_stats"])
                    state_dict[i][j]["anomaly_scores"]=LivePercentile(state_dict[i][j]["anomaly_scores"])
                    if self.use_edge_attr:
                        state_dict[i][j]["edge_stats"]=LivePercentile(state_dict[i][j]["edge_stats"])

            setattr(self, i, state_dict[i])
            del state_dict[i]
            

class MultiLayerGNNIDS(BaseOnlineODModel,EnsembleSaveMixin):
    def __init__(self,  model_name="MultiLayerGNNIDS", n_features=15, l_features=35,
            preprocessors=[], **kwargs):
        BaseOnlineODModel.__init__(self,
            model_name=model_name, n_features=n_features, l_features=l_features,
            preprocessors=preprocessors, **kwargs
        )
        self.protocol_map={"UDP":0} #,"ARP":2,"ICMP":3,"Other":4 "TCP":1
        self.protocol_inv={v:k for k,v in self.protocol_map.items()}
        
        self.ensemble_models=[f"HomoGNNIDS_{k}" for k,v in self.protocol_map.items()]
        self.layer_map={v:HomoGNNIDS(model_name=f"HomoGNNIDS_{k}", **kwargs) for k,v in self.protocol_map.items()}
        self.processed=0
        self.batch_num=0
        self.draw_graph=[False for i in range(len(self.ensemble_models))]
        
    def process(self, x):        
        results_list=[]
        anomaly=0
        for protocol, model in self.layer_map.items():
            
            idx=torch.where(x[:,3].long()==protocol)
            data=x[idx]
            
            if data.nelement()>0:
                
                results=model.process(data)
                
                if results is not None:    
                    self.draw_graph[protocol]=True
                    
                    results["protocol"]=self.protocol_inv[protocol]
                    results["index"]=self.batch_num
                    
                    results_list.append(results)
                    if (results["loss"]>results["threshold"]).any():
                        anomaly=1
        self.batch_num+=1        
        self.processed+=x.size(0)
        
        if len(results_list)==0:
            return None, None
        
        return {"anomaly":anomaly}, results_list
            
    def visualize_graph(self, dataset_name, fe_name, file_name):
        n_plots=np.sum(self.draw_graph)
        if n_plots==0:
            return
        
        fig, ax = plt.subplots(figsize=(8, n_plots*5), nrows=n_plots, squeeze=False)
        
        node_map=get_node_map(fe_name, dataset_name)
        if node_map is not None:
            node_map={v:k for k,v in node_map.items()}
        
        ax_idx=0
        for protocol, model in self.layer_map.items():
            if self.draw_graph[protocol]:
                ax[ax_idx][0].set_title(f"{self.protocol_inv[protocol]} after {model.processed} packets")
                if model.G.x.nelement()>0:
                    model.draw_graph(fig, ax[ax_idx][0], node_map)
                else:
                    ax[ax_idx][0].annotate("Empty Graph", (0,0))
                ax_idx+=1
                self.draw_graph[protocol]=False 
                
        fig.tight_layout()
        path=Path(f"../../datasets/{dataset_name}/{fe_name}/graph_plots/{file_name}/{self.model_name}_{self.processed}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format='png')
        # Close the figure to release resources
        plt.close(fig)
            
    
        