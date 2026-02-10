from abc import abstractmethod
from typing import List
from ..converters.types import RecordList
from ..converters.decorator import auto_cast_method
from ..save_mixin.pickle import PickleSaveMixin
from ..component.feature_extractor import BaseFeatureExtractor
from scapy.all import (
    Ether,
    ARP,
    IP,
    IPv6,
    TCP,
    UDP,
    ICMP,
    Raw,
    DNS,
    DNSQR,
    DNSRR,  # for DNS info if needed
    rdpcap,
)
import networkx as nx
import pandas as pd


class GraphExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, granularity=1,**kwargs):
        super().__init__(**kwargs)
        self.granularity = granularity
        self.time_stamp = None
        self.G=self.init_graph()
        
    @abstractmethod 
    def init_graph(self):
        pass
        
    def export_graph(self):
        self.G.graph["time_stamp"] = self.time_stamp
        data = nx.readwrite.json_graph.node_link_data(self.G)
        self.G=self.init_graph()
        return data
    
    @auto_cast_method
    def update(self, traffic_vectors:RecordList)->List:
        # remove possible None values
        return [x for x in (self.update_single(i) for i in traffic_vectors) if x is not None]


    @abstractmethod
    def update_single(self, traffic_vector):
        pass 

class SingleLayerGraphExtractor(GraphExtractor):
    def __init__(self,  layer="physical",**kwargs):
        """"""
        super().__init__(**kwargs)
        self.layer=layer


    def init_graph(self):
        return nx.DiGraph()

    def peek(self, traffic_vectors):
        pass



    def setup(self):
        pass 
    
    def teardown(self):
        pass

    @property
    def headers(self):
        return ["size", "count"] # header for nodes

    def update_single(self, traffic_vector):

        time_stamp = traffic_vector["timestamp"]
        length = traffic_vector['packet_size']
        
        if self.layer=="physical":
            src = traffic_vector['srcMAC']
            dst = traffic_vector['dstMAC']
        elif self.layer=="internet":
            src=traffic_vector['srcIP']
            dst=traffic_vector['dstIP']
        elif self.layer=="transport":
            
            src=f"{traffic_vector['srcIP']}:{traffic_vector['srcport']}"
            dst=f"{traffic_vector['dstIP']}:{traffic_vector['dstport']}"
        else:
            raise ValueError("Unknown layer choice")

        if not self.G.has_node(src):
            self.G.add_node(src, count=0, size=0)

        if not self.G.has_node(dst):
            self.G.add_node(dst, count=0, size=0)

        self.G.nodes[src]["count"] -= 1
        self.G.nodes[src]["size"] -= length

        self.G.nodes[dst]["count"] += 1
        self.G.nodes[dst]["size"] += length

        self.G.add_edge(src, dst)
        
        if self.time_stamp is None:
            self.time_stamp = time_stamp
            
        
        if time_stamp - self.time_stamp > self.granularity:
            self.time_stamp = time_stamp
            return self.export_graph()
        

    def __str__(self):
        return f"SingleLayerGraphExtractor({self.layer})"