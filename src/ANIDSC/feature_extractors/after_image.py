import numpy as np
from pprint import pformat
import copy
from scapy.all import IP, IPv6, TCP, UDP, ARP, ICMP, Packet
from socket import getservbyport
from itertools import product
from typing import List, Dict, Any
from numpy.typing import NDArray
from ..base_files import PickleSaveMixin, BaseTrafficFeatureExtractor

class AfterImage(BaseTrafficFeatureExtractor, PickleSaveMixin):
    def __init__(self, decay_factors:List[float]=[5, 3, 1, 0.1, 0.01], **kwargs):
        """initializes afterimage, a packet-based feature extractor used in Kitsune

        Args:
            decay_factors (list, optional): the time windows. Defaults to [5,3,1,.1,.01].
        """
        super().__init__(**kwargs)
        self.decay_factors = decay_factors
        self.clean_up_round=5000
            
    def init_state(self):
        """sets up IncStatDB"""
        
        self.state = IncStatDB(
                    decay_factors=self.decay_factors,
                    limit=float("inf"),
                )
    
    def peek(self, traffic_vectors:List[Dict[str, Any]]):
        """fake update. obtains a copy of existing database,
        applies the traffic vectors to it.

        Args:
            traffic_vectors (2d array): list of traffic vectors to be updated

        Returns:
            2d array: the corresponding features
        """
        if isinstance(traffic_vectors, dict):
            traffic_vectors=[traffic_vectors]
        
        fake_db = IncStatDB(self.decay_factors)
        for traffic_vector in traffic_vectors:
            
            # srcMAC-IP
            fake_db.stat1d[f"{traffic_vector['srcMAC']}_{traffic_vector['srcIP']}"] = copy.deepcopy(
                self.state.stat1d.get(f"{traffic_vector['srcMAC']}_{traffic_vector['srcIP']}")
            )  
            
            # jitter
            fake_db.stat1d[f"{traffic_vector['srcIP']}_{traffic_vector['dstIP']}"] = copy.deepcopy(
                self.state.stat1d.get(f"{traffic_vector['srcIP']}_{traffic_vector['dstIP']}")
            )  

            # channel
            fake_db.stat1d[f"{traffic_vector['srcIP']}"] = copy.deepcopy(
                self.state.stat1d.get(f"{traffic_vector['srcIP']}")
            )
            fake_db.stat1d[f"{traffic_vector['dstIP']}"] = copy.deepcopy(
                self.state.stat1d.get(f"{traffic_vector['dstIP']}")
            )
            fake_db.stat2d[f"{traffic_vector['srcIP']}->{traffic_vector['dstIP']}"] = copy.deepcopy(
                self.state.stat2d.get(f"{traffic_vector['srcIP']}->{traffic_vector['dstIP']}")
            )

            # socket
            if traffic_vector['srcport'] =="ARP":
                fake_db.stat1d[f"{traffic_vector['srcMAC']}"] = copy.deepcopy(
                    self.state.stat1d.get(f"{traffic_vector['srcMAC']}")
                )
                fake_db.stat1d[f"{traffic_vector['dstMAC']}"] = copy.deepcopy(
                    self.state.stat1d.get(f"{traffic_vector['dstMAC']}")
                )
                fake_db.stat2d[f"{traffic_vector['srcMAC']}->{traffic_vector['dstMAC']}"] = copy.deepcopy(
                    self.state.stat2d.get(f"{traffic_vector['srcMAC']}->{traffic_vector['dstMAC']}")
                )
            else:
                fake_db.stat1d[f"{traffic_vector['srcIP']}_{traffic_vector['srcport']}"] = copy.deepcopy(
                    self.state.stat1d.get(f"{traffic_vector['srcIP']}_{traffic_vector['srcport']}")
                )
                fake_db.stat1d[f"{traffic_vector['dstIP']}_{traffic_vector['dstport']}"] = copy.deepcopy(
                    self.state.stat1d.get(f"{traffic_vector['dstIP']}_{traffic_vector['dstport']}")
                )
                fake_db.stat2d[f"{traffic_vector['srcIP']}_{traffic_vector['srcport']}->{traffic_vector['dstIP']}_{traffic_vector['dstport']}"] = copy.deepcopy(
                    self.state.stat2d.get(f"{traffic_vector['srcIP']}_{traffic_vector['srcport']}->{traffic_vector['dstIP']}_{traffic_vector['dstport']}")
                )

        fake_db.num_updated = self.state.num_updated
        fake_db.num_entries = self.state.num_entries
        fake_db.last_timestamp=self.state.last_timestamp
        fake_db.mac_to_idx_map=self.state.mac_to_idx_map
        
        vectors = []
        for tv in traffic_vectors:
            vectors.append(self.update(tv, fake_db))
        return np.vstack(vectors)


    def update(self, traffic_vector:Dict[str, Any], state=None):
        """updates the internal state with traffic vector

        Args:
            traffic_vector (array): a traffic vectors consists of
            data extracted from pacekts

        Returns:
            array: the extracted features
        """
        if state is None:
            state=self.state
        
        src_mac_ip = state.update_get_stats_1D(
            f"{traffic_vector['srcMAC']}_{traffic_vector['srcIP']}", traffic_vector['timestamp'], traffic_vector["packet_size"]
        )  
        
         # jitter between channels
        jitter = state.update_get_stats_1D(
            f"{traffic_vector['srcIP']}_{traffic_vector['dstIP']}", traffic_vector['timestamp'], None
        )  
        
        # channel: sent between this packet’s source and destination IPs
        channel = state.update_get_stats_2D(
            f"{traffic_vector['srcIP']}",f"{traffic_vector['dstIP']}", traffic_vector['timestamp'], traffic_vector["packet_size"]
        )  
        

        # Socket: sent between this packet’s source and destination TCP/UDP Socket
        # arp has no IP
        if traffic_vector['srcport'] =="ARP":
            socket = state.update_get_stats_2D(
                f"{traffic_vector['srcMAC']}", f"{traffic_vector['dstMAC']}", traffic_vector['timestamp'],
                traffic_vector["packet_size"]
            )
        else:
            socket = state.update_get_stats_2D(
                f"{traffic_vector['srcIP']}_{traffic_vector['srcport']}",
                f"{traffic_vector['dstIP']}_{traffic_vector['dstport']}",
                traffic_vector['timestamp'], traffic_vector["packet_size"]
            )
        
        state.num_updated+=1 
        
        state.last_timestamp=traffic_vector["timestamp"]
        
        feature=np.hstack([src_mac_ip, jitter, channel, socket])
        
        # clean our records
        if state.num_updated % self.clean_up_round == 0:
            state.clean_records(traffic_vector['timestamp'])

        return np.expand_dims(feature, axis=0)

    def get_traffic_vector(self, packet:Packet):
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

        
        traffic_vector = {"srcMAC":srcMAC,
                          "dstMAC":dstMAC,
                            "srcIP":srcIP,
                            "srcport":srcproto,
                            "dstIP":dstIP,
                            "dstport":dstproto,
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
        stream_1d = ["Src-MAC-IP", "Jitter"]
        stream_2d = ["Channel", "Socket"]
        headers = []
        for name, stat in product(stream_1d, stat_1d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        for name, stat in product(stream_2d, stat_1d + stat_2d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")

        return headers

    def get_meta_headers(self):
        """return the feature names of traffic vectors

        Returns:
            list: names of traffic vectors
        """
        return [
            "ip_type",
            "src_mac",
            "dst_mac",
            "scr_ip",
            "src_protocol",
            "dst_ip",
            "dst_protocol",
            "time_stamp",
            "packet_size",
        ]
        


class AfterImageGraph(AfterImage):
    def __init__(self, protocols:List[str]=["TCP", "UDP", "ARP", "ICMP"], **kwargs):
        """initializes afterimage, a packet-based feature extractor used in Kitsune

        Args:
            limit (int, optional): maximum number of records. Defaults to 1e6.
            decay_factors (list, optional): the time windows. Defaults to [5,3,1,.1,.01].
        """
        super().__init__(skip=4,**kwargs)
        protocols+=["Other"]
        self.protocol_map={p:i for i,p in enumerate(protocols)}
        self.name=self.__str__()
    
    def __str__(self):
        return f"AfterImageGraph({','.join(self.protocol_map.keys())})"
    
    def setup(self):
        super().setup()
        self.parent.context['protocols']=self.protocol_map
        

    def peek(self, traffic_vectors:List[Dict[str, Any]]):
        """fake update. obtains a copy of existing database,
        applies the traffic vectors to it.

        Args:
            traffic_vectors (2d array): list of traffic vectors to be updated

        Returns:
            2d array: the corresponding features
        """
        raise NotImplementedError


    def update(self, traffic_vector:Dict[str,Any]):
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
        protocol=self.protocol_map.get(traffic_vector['protocol'], len(self.protocol_map)-1)
               
        #udpate link stats
        srcID=f"{srcMAC}/{protocol}"
        dstID=f"{dstMAC}/{protocol}"

        
        #update node stats
        src_stat=self.state.update_get_stats_1D(srcID, traffic_vector['timestamp'], traffic_vector['packet_size'])
        dst_stat=self.state.update_get_stats_1D(dstID, traffic_vector['timestamp'], -traffic_vector['packet_size'])
        
        #get link stats
        link_stat=self.state.update_get_stats_2D(srcID, dstID, traffic_vector['timestamp'], traffic_vector['packet_size'], return_1d=False)
        jitter_stat=self.state.update_get_stats_1D(f"{srcID}-{dstID}",traffic_vector['timestamp'], None)
        
        self.state.num_updated += 1

        #encode ID and traffic_vector 
        srcID=self.state.mac_to_idx_map.setdefault(srcMAC, len(self.state.mac_to_idx_map))
        dstID=self.state.mac_to_idx_map.setdefault(dstMAC, len(self.state.mac_to_idx_map))
        
        feature=np.hstack([1, srcID, dstID, protocol, src_stat, dst_stat, jitter_stat, link_stat])
        
        self.state.last_timestamp=traffic_vector["timestamp"]
        self.parent.context['mac_to_idx_map']=self.state.mac_to_idx_map
        
        # clean our records
        if self.state.num_updated % self.clean_up_round == 0:
            n, keys=self.state.clean_records(traffic_vector['timestamp'])
            all_records=[feature]
            for key in keys:
                src, dst=key.split("->")
                srcIP=src.split("/")[0]
                dstIP, protocol=dst.split("/")
                all_records.append([0,self.state.mac_to_idx_map[srcIP],self.state.mac_to_idx_map[dstIP],int(protocol)]+[0 for _ in range(65)])

            return np.vstack(all_records)
        else:
            return np.expand_dims(feature,axis=0)

    def get_traffic_vector(self, packet:Packet)->Dict[str, Any]:
        """adds protocol name to traffic vector from after image

        Args:
            packet (scapy packet): input packet

        Returns:
            Dict[str, Any]: dictionary of traffic vector
        """
        traffic_vector=super().get_traffic_vector(packet)

        if traffic_vector is None:
            return None 
        
        if TCP in packet:
            protocol=self.get_protocol_name(packet.sport, packet.dport, "tcp")
        elif UDP in packet:
            protocol=self.get_protocol_name(packet.sport, packet.dport, "udp")
        elif ARP in packet:
            protocol="ARP"
        elif ICMP in packet:
            protocol="ICMP"
        else:
            protocol="Other"
        
        traffic_vector["protocol"]= protocol

        return traffic_vector

    def get_protocol_name(self, src_port_num: int, dst_port_num:int , proto:str)->str:
        """gets the protocol name associated with port number

        Args:
            src_port_num (int): source port number 
            dst_port_num (int): destination port number
            proto (str): protocol, either 'udp' or 'tcp'

        Returns:
            str: protocol string for the port
        """        
        try:
            protocol=getservbyport(src_port_num,proto)
        except OSError:
            try:
                protocol=getservbyport(dst_port_num,proto)
            except OSError:
                protocol=proto
        
        if protocol not in self.protocol_map.keys():
            protocol=proto
            
        return protocol.upper()
    
    def get_headers(self)->List[str]:
        """returns the feature names

        Returns:
            List[str]: list of feature names
        """

        stat_1d = ["weight", "mean", "std"]

        stat_2d = ["magnitude", "radius", "covariance", "pcc"]
        stream_1d = ["src", "dst", "jitter"]
        headers = ["type","srcID","dstID","protocol"]
        for name, stat in product(stream_1d, stat_1d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        for name, stat in product(['link'], stat_2d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        return headers
    
    def get_meta_headers(self)->List[str]:
        """return the feature names of traffic vectors

        Returns:
            List[str]: names of traffic vectors
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

def magnitude(x:float, y:float):
    """the magnitude of a set of incStats, pass var instead of mean to get radius

    Args:
        x (float): first value
        y (float): second value

    Returns:
        float: result
    """
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


class IncStat1D:
    def __init__(self, len_factors: int):  # timestamp is creation time
        """
        Incremental Statistics for 1 dimensional features
        keeps an array of size 3*decay_factors.
        The array being: [[weight1, weight2, ..., weightn],
                            [sum1, sum2, ..., sumn],
                            [squared sum1, squared sum2, ..., squared sum3]]

        Args:
            decay_factors (array): array of decay factors (time windows)
            name (string): name of this IncStat
            t (float): timestamp of this incstat
            v (float, optional): value of this incstat. pass None to use jitter. Defaults to None.
        """
        self.weight_thresh = 1e-3
        self.last_timestamp=None
        self.incremental_statistics=None
        self.len_factors=len_factors

    
    def __repr__(self):
        return pformat(vars(self))
    
    def insert(self, decay_factors, t:float, v =None):
        """updates the incstat with value.
        decays the statistics and increments it.

        Args:
            t (float): time of new information
            v (float, optional): value of new information. use None to measure
            jitter. Defaults to None.
        """
        
        if v is None:
            if self.last_timestamp is None:
                v=0
            else:
                v = t - self.last_timestamp
        
        if self.incremental_statistics is None:
            
            self.incremental_statistics = np.tile(
                np.expand_dims([1.0, v, v**2], axis=1), [1, self.len_factors]
            )  # each row corresponds to weight, linear sum, sum of squares
        
        else:    
            self.decay(decay_factors, t)

            # update with v
            self.incremental_statistics += np.expand_dims(
                [1.0, v, v**2], axis=1
            )  # broadcast to [3,1]
        
        self.last_timestamp = t
            

    def decay(self, decay_factors:NDArray[np.float_], t:float):
        """decays the incremental statistics according to t

        Args:
            t (float): current timestamp
        """
        # check for decay
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2.0 ** (-decay_factors * time_diff)
            self.incremental_statistics *= factor

    def is_outdated(self, decay_factors:NDArray[np.float_], t:float):
        """checks if incstat is outdated.
        if the weights in all time windows are lower than the weight threshold after
        decaying the record, return True

        Args:
            t (float): current timestamp

        Returns:
            boolean: whether the incstat is outdated
        """
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2 ** (-decay_factors * time_diff)
        else:
            return False

        return np.all((self.incremental_statistics[0] * factor) < self.weight_thresh)

    def weight(self):
        """returns the weight of each time window

        Returns:
            array: the weights
        """
        return self.incremental_statistics[0]

    def mean(self):
        """mean of each time window, calculated by sum/weight

        Returns:
            array: mean of each time window
        """
        return np.where(
            self.weight() < self.weight_thresh,
            0.0,
            self.incremental_statistics[1] / self.incremental_statistics[0],
        )

    def var(self):
        """variance of each timewindow. caculated by squared sum/weight - mean **2.
        since the

        Returns:
            array: variance of each timewindow
        """
        return np.where(
            self.weight() < self.weight_thresh,
            0.0,
            np.abs(self.incremental_statistics[2] / self.weight() - self.mean() ** 2),
        )

    def std(self):
        """standard deviation, calculated by sqrt(var)

        Returns:
            array: standard deviation
        """
        return np.sqrt(self.var())

    def all_stats_1D(self):
        """returns all the 1d stats

        Returns:
            array: [weight, mean, std]
        """
        return np.hstack([self.weight(), self.mean(), self.std()])


    
class IncStat2D:
    def __init__(self, len_factor:int):
        """2 dimensional IncStat, stores the relationship between two
        1d inc stat. keeps track of the sum of residual products (A-uA)(B-uB)

        Args:
            incS1 (IncStat): first incstat
            incS2 (IncStat): second incstat
            t (float): current timestamp
        """
        # store references to the streams' incStats
        self.sum_of_residual = np.zeros(len_factor)
        self.eps = 2e-3
        self.last_timestamp = None

    def update_cov(self, inc_stats1, inc_stats2, decay_factors, t:float, v:int):
        """updates the covariance of the two streams.
        decays the residual then updates it.

        Args:
            t (float): current time
            v (float): the value to be updated
        """
        
        if self.last_timestamp is None:
            self.last_timestamp=t 
        else:
            # Decay residules
            self.decay(decay_factors, t)

            # Compute and update residule
            self.sum_of_residual += (v - inc_stats1.mean()) * (
                v - inc_stats2.mean()
            )
            self.last_timestamp = t

    def decay(self,decay_factors:NDArray[np.float_], t:float):
        """decays the residual product. ignores if time diff is negative

        Args:
            t (float): current timestamp
        """
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2 ** (-decay_factors * time_diff)
            self.sum_of_residual *= factor

    def cov(self, inc_stats1, inc_stats2):
        """calculates the covariance between two incstats:
        sum of redsidual/(w1+w2). if weight sum is too small, it will return 0

        Returns:
            array: covariance of each time window
        """
        weight_sum = inc_stats1.weight() + inc_stats2.weight()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                weight_sum < self.eps, 0.0, self.sum_of_residual / weight_sum
            )

    def pcc(self, inc_stats1, inc_stats2):
        """pearson correlation coefficient. calculated via cov/(std1*std2)

        Returns:
            float: pcc of each timestamp
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ss = inc_stats1.std() * inc_stats2.std()
            return np.where(ss < self.eps, 0.0, self.cov(inc_stats1, inc_stats2) / ss)

    def all_stats_2D(self, inc_stats1, inc_stats2):
        """calculates all 2d statistics

        Returns:
            array: [magnitude, radius, cov, pcc]
        """
        mean1 = inc_stats1.mean()
        mean2 = inc_stats2.mean()
        var1 = inc_stats1.var()
        var2 = inc_stats2.var()
        return np.hstack(
            [magnitude(mean1, mean2), magnitude(var1, var2), self.cov(inc_stats1, inc_stats2), self.pcc(inc_stats1, inc_stats2)]
        )
        

    def __repr__(self):
        return pformat(vars(self))

class IncStatDB:
    def __init__(self, decay_factors:List[float], last_timestamp:float=None, limit:int=1e5, mac_to_idx_map:Dict[str, int]={}):
        """database to store all incstats

        Args:
            decay_factors (array): time windows
            limit (int, optional): maximum size of database. Defaults to 1e5.
        """
        # list of dictionary to store 1d stats for each lambda, index matches lambda id
        self.stat1d = {}
        
        # list of dict to store 2d stats for each lambda, index matches lambda id
        self.stat2d = {}
        
        # limit for all lambdas combined
        self.limit = limit
        self.num_entries = 0
        self.decay_factors = np.array(decay_factors)

        # number of pkts updated
        self.num_updated = 0
        self.last_timestamp=last_timestamp
        self.mac_to_idx_map=mac_to_idx_map
    
    def __repr__(self):
        return pformat(vars(self))
    
    def get_stats_1D(self, ID:str):
        if ID not in self.stat1d or self.stat1d[ID] is None:
            if self.num_entries + 1 > self.limit:
                raise LookupError(
                    "Adding Entry:\n"
                    + ID
                    + "\nwould exceed incStat 1D limit of "
                    + str(self.limit)
                    + ".\nObservation Rejected."
                )

            self.stat1d[ID] = IncStat1D(len(self.decay_factors))

            self.num_entries += 1
        
        return self.stat1d[ID]
            
    def update_get_stats_1D(self, ID:str, t:float, v:int):
        """Updates 1d incstat with ID given time and value.
        if ID does not exist, create it.
        Once updated, return the 1d features associated with it.

        Args:
            ID (str): ID of incstat
            t (float): timestamp
            v (float): value

        Raises:
            LookupError: if exceeding the max limit

        Returns:
            array: array of 1d stats
        """
        # not in our db
        stat1d=self.get_stats_1D(ID)
        stat1d.insert(self.decay_factors, t, v)

        return stat1d.all_stats_1D()
    
    def get_stats_2D(self, ID1:str, ID2:str ):
        # check for pre-exiting link
        if (
            f"{ID1}->{ID2}" not in self.stat2d
            or self.stat2d[f"{ID1}->{ID2}"] is None
        ):
            # Link incStats
            self.stat2d[f"{ID1}->{ID2}"] = IncStat2D(len(self.decay_factors))
        return self.stat2d[f"{ID1}->{ID2}"]
    
    def update_get_stats_2D(self, ID1:str, ID2:str, t:float, v:int, return_1d=True):
        """updates incstat of ID1 with t, v and ID2 with t, -v.
        create 2d incstat if it doesnt exist and updates it.
        returns the 1d statistics associated with the sender and the 2d statistics

        Args:
            ID1 (str): ID of incstat1
            ID2 (str): ID of incstat2
            t (float): current timestamp
            v (float): value

        Returns:
            array: 1d+2d statistics
        """
        # for after image graph, theres no need to update 1d stats
        if return_1d:
            # update 1d after covariance
            stats1d = self.update_get_stats_1D(ID1, t, v)

            # the destination receives negative value
            self.update_get_stats_1D(ID2, t, -v)

        # check for pre-exiting link
        stat2d=self.get_stats_2D(ID1, ID2)
        stat2d.update_cov(self.stat1d[ID1], self.stat1d[ID2], self.decay_factors, t, v)
        feature2d=stat2d.all_stats_2D(self.stat1d[ID1], self.stat1d[ID2])

        if return_1d:
            return np.hstack([stats1d, feature2d])
        else:
            return feature2d
        
    def clean_records(self, t:float):
        """cleans out records that have small weight

        Args:
            t (float): current timestamp

        Returns:
            int: number of removed records and records looked through
        """
        removed = 0
        removed_keys=[]
        for key, inc_stat in dict(self.stat1d).items():
            if inc_stat.is_outdated(self.decay_factors, t):
                # remove 2d links
                for cov_key in dict(self.stat2d).keys():
                    if key in cov_key:
                        del self.stat2d[cov_key]
                        removed_keys.append(cov_key)
                # remove self
                del self.stat1d[key]
                removed += 1
        return removed, removed_keys

    