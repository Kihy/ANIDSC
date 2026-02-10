from ..feature_extractor.simple import GraphExtractor
from ..save_mixin.pickle import PickleSaveMixin
from ..component.feature_extractor import BaseFeatureExtractor
from typing import List
from ..converters.types import RecordList
from ..converters.decorator import auto_cast_method
import networkx as nx
import pandas as pd

class MultiLayerGraphExtractor(GraphExtractor):
    def teardown(self):
        """Cleanup"""
        pass
    
    def setup(self):
        pass 
    
    def peek(self):
        pass
    
    @property
    def headers(self):
        return ["size", "count", "layer"]

    def _update_connection(self, src, dst, length):
        """Update edge between two nodes"""
        if self.G.has_edge(src, dst):
            self.G[src][dst]["size"] += length
            self.G[src][dst]["count"] += 1
        else:
            self.G.add_edge(src, dst, size=length, count=1)

    def init_graph(self):
        return nx.DiGraph()

    def update_single(self, traffic_vector):
        """Process a single packet and build multi-layer graph"""
        time_stamp = traffic_vector['timestamp']
        
        
        src_mac = traffic_vector['srcMAC']
        dst_mac = traffic_vector['dstMAC']
        
        # Add nodes with layer attribute
        if not self.G.has_node(src_mac):
            self.G.add_node(src_mac, layer="Physical")
        if not self.G.has_node(dst_mac):
            self.G.add_node(dst_mac, layer="Physical")
        
        # Get Ethernet payload length
        ether_len = getattr(traffic_vector, 'ether_payload_len', traffic_vector['packet_size'])
        
        # Update horizontal connection at Physical layer
        self._update_connection(src_mac, dst_mac, ether_len)
            

        # Extract IP layer (Internet)
        src_ip = traffic_vector['srcIP']
        dst_ip = traffic_vector['dstIP']
        
        # Add nodes with layer attribute
        if not self.G.has_node(src_ip):
            self.G.add_node(src_ip, layer="Internet")
        if not self.G.has_node(dst_ip):
            self.G.add_node(dst_ip, layer="Internet")
        
        # Get IP payload length
        ip_len = getattr(traffic_vector, 'ip_payload_len', traffic_vector['packet_size'])
        
        # Update horizontal connection at Internet layer
        self._update_connection(src_ip, dst_ip, ip_len)
        
        
        # Add interlayer connections (MAC to IP)
        self._update_connection(src_ip, src_mac, ip_len)
        self._update_connection(dst_mac, dst_ip, ether_len)

        # Extract Transport layer
        port_src = traffic_vector['srcport']
        port_dst = traffic_vector['dstport']
        
        if port_src and port_dst:
            src_ip = traffic_vector['srcIP']
            dst_ip = traffic_vector['dstIP']
            src_socket = f"{src_ip}:{port_src}"
            dst_socket = f"{dst_ip}:{port_dst}"
            
            # Add nodes with layer attribute
            if not self.G.has_node(src_socket):
                self.G.add_node(src_socket, layer="Transport")
            if not self.G.has_node(dst_socket):
                self.G.add_node(dst_socket, layer="Transport")
            
            # Get transport payload length
            transport_len = getattr(traffic_vector, 'transport_payload_len', 
                                    getattr(traffic_vector, 'payload_size', 0))
            
            # Update horizontal connection at Transport layer
            self._update_connection(src_socket, dst_socket, transport_len)
            
            # Update interlayer connections
            self._update_connection(src_socket, src_ip, transport_len)
            self._update_connection(dst_ip, dst_socket, ip_len)
        
        # Initialize timestamp
        if self.time_stamp is None:
            self.time_stamp = time_stamp
        
        # Check if we need to export graph

        if time_stamp - self.time_stamp > self.granularity:
            graph = self.export_graph()
            self.time_stamp = time_stamp
            return graph
        

    def __str__(self):
        return f"MultiLayerGraphExtractor({self.granularity})"