from ANIDSC.feature_buffer.json import JsonFeatureBuffer
from ANIDSC.feature_buffer.tabular import NumpyFeatureBuffer
import numpy as np
from networkx.readwrite import json_graph
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


class SingleLayerGraphExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, granularity=1, layer="physical",**kwargs):
        """"""
        super().__init__(**kwargs)
        self.layer=layer
        self.granularity = granularity
        self.time_stamp = None
        self.G = nx.DiGraph()


    def peek(self, traffic_vectors):
        pass

    def export_graph(self):
        self.G.graph["time_stamp"] = self.time_stamp
        data = nx.readwrite.json_graph.node_link_data(self.G)
        self.G = nx.DiGraph()
        return data

    def setup(self):
        pass 
    
    def teardown(self):
        pass

    @property
    def headers(self):
        return ["size", "count"] # header for nodes
    
    def update(self, traffic_vector):
        if isinstance(traffic_vector, pd.DataFrame):
            ret=[]
            for row in traffic_vector.itertuples(index=False):
                result=self.update_single(row)
                if result is not None:
                    ret.append(result)
            if len(ret)==0:
                return None 
            else:
                return ret

        else:
            return self.update_single(traffic_vector)

    def update_single(self, traffic_vector):

        time_stamp = traffic_vector.timestamp
        length = traffic_vector.packet_size
        
        if self.layer=="physical":
            src = traffic_vector.srcMAC
            dst = traffic_vector.dstMAC
        elif self.layer=="internet":
            src=traffic_vector.srcIP
            dst=traffic_vector.dstIP
        elif self.layer=="transport":
            
            src=f"{traffic_vector.srcIP}:{traffic_vector.srcport}"
            dst=f"{traffic_vector.dstIP}:{traffic_vector.dstport}"
        else:
            raise ValueError("Unknown layer choice")

        if self.time_stamp is None:
            self.time_stamp = time_stamp
            
        if time_stamp - self.time_stamp > self.granularity:
            self.time_stamp = time_stamp
            graph = self.export_graph()
            
        else:
            graph = None

        if not self.G.has_node(src):
            self.G.add_node(src, count=0, size=0)

        if not self.G.has_node(dst):
            self.G.add_node(dst, count=0, size=0)

        self.G.nodes[src]["count"] -= 1
        self.G.nodes[src]["size"] -= length

        self.G.nodes[dst]["count"] += 1
        self.G.nodes[dst]["size"] += length

        self.G.add_edge(src, dst)

        return graph

    def __str__(self):
        return f"SingleLayerGraphExtractor({self.layer})"