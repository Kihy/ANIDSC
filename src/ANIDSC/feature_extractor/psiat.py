
import numpy as np

from ..save_mixin.pickle import PickleSaveMixin
from ..component.feature_extractor import BaseTrafficFeatureExtractor
from scapy.all import (
    Ether, ARP, IP, IPv6, TCP, UDP, ICMP, Raw,
    DNS, DNSQR, DNSRR, # for DNS info if needed
    rdpcap
)
import networkx as nx

class SimpleState:
    def __init__(self):
        self.last_timestamp=None
        self.last_graph_timestamp=None
        self.graphs={"connectivity":nx.Graph()}
        self.time_window=10
        self.node_features=["sent_iat","sent_size","received_iat","received_size"]
        self.clear_graph=False
        
    def update(self, traffic_vector):
        if self.clear_graph:
            self.graphs={"connectivity":nx.Graph()}
            self.clear_graph=False
        
        
        layers=traffic_vector["features"]
        
        src_root=f"{layers[0][0]}-{layers[0][1]}" 
        dst_root=f"{layers[0][0]}-{layers[0][2]}" 
        
        if src_root not in self.graphs:
            self.graphs[src_root]=nx.Graph()
        
        if dst_root not in self.graphs:
            self.graphs[dst_root]=nx.Graph()
            
        if src_root not in self.graphs["connectivity"]:
            self.graphs["connectivity"].add_node(src_root)
            
        if dst_root not in self.graphs["connectivity"]:
            self.graphs["connectivity"].add_node(dst_root)
            
        self.graphs["connectivity"].add_edge(src_root,dst_root)
        
        src_G=self.graphs[src_root]
        dst_G=self.graphs[dst_root]
        
        # device_graph
        prev_src=None 
        prev_dst=None
        for protocol, src_id, dst_id, iat, size in layers:
            src_node=f"{protocol}-{src_id}"
            dst_node=f"{protocol}-{dst_id}"
            
            # Ensure each node exists in the graph
            if src_node not in src_G:
                src_G.add_node(src_node)
            if dst_node not in dst_G:
                dst_G.add_node(dst_node)
            
            src_G.nodes[src_node].setdefault("sent_iat",[]).append(iat)
            src_G.nodes[src_node].setdefault("sent_size",[]).append(size)
            
            dst_G.nodes[dst_node].setdefault("received_iat",[]).append(iat)
            dst_G.nodes[dst_node].setdefault("received_size",[]).append(size)
            
            if prev_src is not None:
                src_G.add_edge(prev_src, src_node)    
                dst_G.add_edge(prev_dst, dst_node)
            
            prev_src=src_node 
            prev_dst=dst_node 
        
        
        
        if traffic_vector["timestamp"] - self.last_graph_timestamp >= self.time_window:
            for root, graph in self.graphs.items():
                for node in graph.nodes():
                    for feature in self.node_features:
                        graph.nodes[node][feature]=np.median(graph.nodes[node].get(feature, [-1]))
            self.clear_graph=True
            self.last_graph_timestamp=traffic_vector["timestamp"]
            return self.graphs
                 
    

class PacketSizeIATExtractor(PickleSaveMixin, BaseTrafficFeatureExtractor):
    def __init__(self, time_window=10, **kwargs):
        """simple frequency feature extractor based on time windows

        Args:
            time_window (int, optional): length of time window. Defaults to 10.
        """                
        super().__init__(**kwargs)
        self.time_window=time_window
        self.state=SimpleState()
         
    
    def peek(self, traffic_vectors):
        pass
    
   
        
    def get_traffic_vector(self, packet):
        if self.state.last_timestamp is None:
            iat=0
            self.state.last_graph_timestamp=packet.time
        else:
            iat=packet.time-self.state.last_timestamp
            
        self.state.last_timestamp=packet.time

        layer = packet
        features=[]

        while layer:
            layer_feature=[layer.name, "","", float(iat),len(bytes(layer.payload)) if layer.payload else 0]
            
            
            # Identify layer type and extract interesting fields
            if layer.name == "Ethernet" and isinstance(layer, Ether):
                # MAC addresses
                layer_feature[1] = layer.src
                layer_feature[2] = layer.dst

            elif layer.name == "ARP" and isinstance(layer, ARP):
                # ARP info
                layer_feature[1]  = layer.psrc
                layer_feature[2]  = layer.pdst

            elif layer.name == "IP" and isinstance(layer, IP):
                # IPv4 addresses
                layer_feature[1] = layer.src
                layer_feature[2] = layer.dst

            elif layer.name == "IPv6" and isinstance(layer, IPv6):
                # IPv6 addresses
                layer_feature[1] = layer.src
                layer_feature[2] = layer.dst

            elif layer.name == "TCP" and isinstance(layer, TCP):
                # TCP ports
                layer_feature[1] = layer.sport
                layer_feature[2] = layer.dport

            elif layer.name == "UDP" and isinstance(layer, UDP):
                # UDP ports
                layer_feature[1] = layer.sport
                layer_feature[2] = layer.dport
             
            features.append(layer_feature)

            # Move to the next layer
            layer = layer.payload
        
        return {"timestamp":packet.time, "features":features}
    
    def get_headers(self):
        return ["protocol","src_id","dst_id","iat","size"]

    def get_meta_headers(self):
        return ["timestamp", "features"]
    
    def update(self, traffic_vector):
        
        return self.state.update(traffic_vector)