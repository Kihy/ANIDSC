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


class NetworkAccessGraphExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, granularity=60, **kwargs):
        """"""
        super().__init__(**kwargs)

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
        src_mac = traffic_vector.srcMAC
        dst_mac = traffic_vector.dstMAC

        if self.time_stamp is None:
            self.time_stamp = time_stamp
            
        if time_stamp - self.time_stamp > self.granularity:
            self.time_stamp = time_stamp
            graph = self.export_graph()
            
        else:
            graph = None

        if not self.G.has_node(src_mac):
            self.G.add_node(src_mac, count=0, size=0)

        if not self.G.has_node(dst_mac):
            self.G.add_node(dst_mac, count=0, size=0)

        self.G.nodes[src_mac]["count"] -= 1
        self.G.nodes[src_mac]["size"] -= length

        self.G.nodes[dst_mac]["count"] += 1
        self.G.nodes[dst_mac]["size"] += length

        self.G.add_edge(src_mac, dst_mac)

        return graph
