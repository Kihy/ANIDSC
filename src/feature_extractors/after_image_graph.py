import numpy as np
from pprint import pformat
import copy
from scapy.all import *
from scapy.layers.http import *
from scapy.layers.inet import *
from .base_feature_extractor import *
from pathlib import Path
import pickle
import json
from utils import *
from tqdm import tqdm
from .after_image import IncStatDB

class AfterImageGraph(BaseTrafficFeatureExtractor):
    def __init__(self, graph_type="homo",decay_factors=[5, 3, 1, 0.1, 0.01], max_pkt=float("inf"), protocols=["UDP","TCP","ARP"], **kwargs):
        """initializes afterimage, a packet-based feature extractor used in Kitsune

        Args:
            limit (int, optional): maximum number of records. Defaults to 1e6.
            decay_factors (list, optional): the time windows. Defaults to [5,3,1,.1,.01].
        """
        super().__init__(**kwargs)
        self.decay_factors = decay_factors
        self.name = f"AfterImageGraph"
        self.graph_type=graph_type
        self.max_pkt =max_pkt
        self.clean_up_round=5000
        protocols+=["Other"]
        self.protocol_map={p:i for i,p in enumerate(protocols)}
        
        if self.state is None:
            self.state={"db": IncStatDB(
                    decay_factors=self.decay_factors,
                    limit=float("inf"),
                ),
                        "node_map":{},
                        "protocol_map":self.protocol_map,
                        "last_timetamp":None 
                            }    

    def setup(self):
        """sets up after image"""
        super().setup()
        if self.reset_state:
            
            self.state={"db": IncStatDB(
                decay_factors=self.decay_factors,
                limit=float("inf"),
            ),
                       "node_map":{},
                       "protocol_map":self.protocol_map,
                       "last_timetamp":None 
                        } 

    def peek(self, traffic_vectors):
        """fake update. obtains a copy of existing database,
        applies the traffic vectors to it.

        Args:
            traffic_vectors (2d array): list of traffic vectors to be updated

        Returns:
            2d array: the corresponding features
        """
        pass

    def extract_features(self):
        """main loop to extract the features. If state is set,
        change the time so that it is starts immediately after the benign
        traffic.
        for each packet, extract the traffic vectors and get features. Write to
        file every 10000 records
        """
        self.setup()

        features_list = []
        meta_list = []
        
        chunk_size=0
        for packet in tqdm(self.input_pcap, desc=f"parsing {self.file_name}"):
            if self.count>self.max_pkt:
                break
            
            traffic_vector = self.get_traffic_vector(packet)
            
            if traffic_vector is None:
                self.skipped += 1
                continue
            
            self.state["last_timestamp"]=traffic_vector["timestamp"]

            if self.offset_timestamp:
                if self.offset_time is None:
                    self.offset_time = traffic_vector["timestamp"] - self.state["last_timestamp"]
                        
                traffic_vector["timestamp"] -= self.offset_time
            
            feature = self.update(traffic_vector)
            features_list.append(feature)
            
            meta_list.append([self.count]+[i for i in traffic_vector.values()])
            
            self.count += feature.shape[0]
            chunk_size+=feature.shape[0]
            
            if chunk_size > 1e4:
                np.savetxt(
                    self.feature_file,
                    np.vstack(features_list),
                    delimiter=",",
                    fmt="%s",
                )
                np.savetxt(
                    self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s"
                )
                features_list = []
                meta_list = []
                chunk_size=0

        # save remaining
        np.savetxt(
            self.feature_file, np.vstack(features_list), delimiter=",", fmt="%s"
        )
        np.savetxt(self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")

        self.teardown()

    def update(self, traffic_vector):
        """updates the internal state with traffic vector

        Args:
            traffic_vector (array): a traffic vectors consists of
            data extracted from pacekts

        Returns:
            array: the extracted features
        """
        srcMAC=f"{traffic_vector['srcMAC']}"
        dstMAC=f"{traffic_vector['dstMAC']}"
        
        #get protocol
        protocol=self.state["protocol_map"].get(traffic_vector['protocol'], 4)
               
        #udpate link stats
        if self.graph_type=="homo":
            srcID=srcMAC
            dstID=dstMAC
            src_link=srcMAC
            dst_link=dstMAC
        elif self.graph_type=="hetero":
            srcID=srcMAC
            dstID=dstMAC
            src_link=f"{srcID}/{protocol}"
            dst_link=f"{dstID}/{protocol}"
        elif self.graph_type=="multi_layer":
            srcID=f"{srcMAC}/{protocol}"
            dstID=f"{dstMAC}/{protocol}"
            src_link=srcID
            dst_link=dstID
        else:
            raise ValueError("Unknow graph_type")
        
        #update node stats
        src_stat=self.state["db"].update_get_stats_1D(srcID, traffic_vector['timestamp'], traffic_vector['packet_size'])
        dst_stat=self.state["db"].update_get_stats_1D(dstID, traffic_vector['timestamp'], -traffic_vector['packet_size'])
        
        #get link stats
        link_stat=self.state["db"].update_get_link_stats(src_link, dst_link, traffic_vector['timestamp'], traffic_vector['packet_size'])
        jitter_stat=self.state["db"].update_get_stats_1D(f"{src_link}-{dst_link}",traffic_vector['timestamp'], None)
        
        self.state["db"].num_updated += 1

        #encode ID and traffic_vector 
        srcID=self.state["node_map"].setdefault(srcMAC, len(self.state["node_map"]))
        dstID=self.state["node_map"].setdefault(dstMAC, len(self.state["node_map"]))
        
        feature=np.hstack([1, srcID, dstID, protocol, src_stat, dst_stat, jitter_stat, link_stat])
        
        # clean our records
        if self.state["db"].num_updated % self.clean_up_round == 0:
            n, keys=self.state["db"].clean_records(traffic_vector['timestamp'])
            all_records=[feature]
            for src,dst in keys:
                if self.graph_type=="homo":
                    all_records.append([0,self.state["node_map"][src],self.state["node_map"][dst],0]+[0 for _ in range(65)])
                else:
                    srcIP=src.split("/")[0]
                    dstIP, protocol=dst.split("/")
                    all_records.append([0,self.state["node_map"][srcIP],self.state["node_map"][dstIP],protocol]+[0 for _ in range(65)])

            return np.vstack(all_records)
        else:
            return np.expand_dims(feature,axis=0)

    def get_traffic_vector(self, packet):
        """extracts the traffic vectors from packet

        Args:
            packet (scapy packet): input packet

        Returns:
            array: list of IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, time, packet size
        """
        packet = packet[0]

        # only process IP packets and non broadcast/localhost or packet.dst in ["ff:ff:ff:ff:ff:ff","00:00:00:00:00:00"]
        if not (packet.haslayer(IP) or packet.haslayer(IPv6) or packet.haslayer(ARP)):
            return None

        timestamp = packet.time
        framelen = len(packet)
        if packet.haslayer(IP):  # IPv4
            srcIP = packet[IP].src
            dstIP = packet[IP].dst
        elif packet.haslayer(IPv6):  # ipv6
            srcIP = packet[IPv6].src
            dstIP = packet[IPv6].dst

        else:
            srcIP = ""
            dstIP = ""

        if packet.haslayer(TCP):
            srcproto = str(packet[TCP].sport)
            dstproto = str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            srcproto = str(packet[UDP].sport)
            dstproto = str(packet[UDP].dport)
        else:
            srcproto = ""
            dstproto = ""

        if packet.haslayer(ARP):
            srcMAC = packet[ARP].hwsrc
            dstMAC = packet[ARP].hwdst
        else:
            srcMAC = packet.src
            dstMAC = packet.dst

        if srcproto == "":  # it's a L2/L1 level protocol
            if packet.haslayer(ARP):  # is ARP
                srcproto = "arp"
                dstproto = "arp"
                srcIP = packet[ARP].psrc  # src IP (ARP)
                dstIP = packet[ARP].pdst  # dst IP (ARP)

            elif packet.haslayer(ICMP):  # is ICMP
                srcproto = "icmp"
                dstproto = "icmp"

            elif srcIP + srcproto + dstIP + dstproto == "":  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        layers=packet.layers()
        
        if layers[-1]._name in ["Raw","Padding"]:
            lowest_layer=layers[-2]._name
        else:
            lowest_layer=layers[-1]._name

        
        traffic_vector = {"srcMAC":srcMAC,
                          "dstMAC":dstMAC,
            "srcIP":srcIP,
            "srcport":srcproto,
            "dstIP":dstIP,
            "dstport":dstproto,
            "protocol": str(lowest_layer),
            "timestamp":float(timestamp),
            "packet_size":int(framelen)}

        return traffic_vector

    def get_headers(self):
        """returns the feature names

        Returns:
            list: list of feature names
        """

        stat_1d = ["weight", "mean", "std"]

        stat_2d = ["magnitude", "radius", "covariance", "pcc"]
        stream_1d = ["src", "dst", "jitter"]
        headers = ["type","srcID","dstID","protocol"]
        for name, stat in itertools.product(stream_1d, stat_1d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        for name, stat in itertools.product(['link'], stat_2d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        return headers
    
    
    def get_meta_headers(self):
        """return the feature names of traffic vectors

        Returns:
            list: names of traffic vectors
        """
        return [
            "idx",
            "srcMAC",
            "dstMAC",
            "srcIP",
            "srcport",
            "dstIP",
            "dstport",
            "protocol",
            "timestamp",
            "packet_size"
        ]