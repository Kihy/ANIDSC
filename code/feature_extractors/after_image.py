import numpy as np
from pprint import pformat
import copy
from scapy.all import *
from scapy.layers.http import *
from scapy.layers.inet import *
from .base import *
from pathlib import Path
import pickle
import json
from utils import *
from tqdm import tqdm

class AfterImage(BaseTrafficFeatureExtractor, LazyInitializationMixin):
    def __init__(self, limit=1e6, decay_factors=[5,3,1,.1,.01], nstat=None, log_file=None, **kwargs):
        self.limit = limit
        self.decay_factors=decay_factors
        self.reset_nstat=True
        self.log_file=log_file
        if nstat is not None:
            self.nstat=nstat
            self.reset_nstat=False
        self.prev_pkt_time=None
        
        self.allowed=("path","nstat")
        self.lazy_init(**kwargs)
        self.entry=self.extract_features
        
    def setup(self):
        self.basename=Path(self.path).stem
        self.curPacketIndx = 0
        if self.reset_nstat:
            self.nstat = NetStat(decay_factors= self.decay_factors, limit = self.limit, log_path=self.log_file)

        self.packets=PcapReader(self.path)

        feature_file=Path(self.path).parents[1]/"features"/(self.basename+".csv")
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file=Path(self.path).parents[1]/"meta"/(self.basename+".csv")
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        self.feature_file=open(feature_file, "w")
        self.meta_file=open(meta_file, "w")
        
        self.count = 0
        self.skipped = 0
                
        

    def peek(self, traffic_vectors):
        first_record=traffic_vectors[0]
        fake_db=self.nstat.get_records(*first_record)
        vectors=[]
        for tv in traffic_vectors:
            vectors.append(self.nstat.update_get_stats(*tv, fake_db))
        return vectors
        
        
    def teardown(self):
        #save netstat information
        netstat_path=Path(self.path).parents[1]/"netstat"/(self.basename+".pkl")
        netstat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(netstat_path, "wb") as pf:
            pickle.dump(self.nstat, pf)
            
            
        self.meta_file.close()
        self.feature_file.close()
        
        #save file information
        data_info=load_dataset_info()
        
        data_info[self.basename]={
            "pcap_path":self.path,
            "feature_path":self.feature_file,
            "meta_path":self.meta_file,
            "num_rows":self.count}
        
        save_dataset_info(data_info)
        print(f"skipped: {self.skipped} written: {self.count}")



    def extract_features(self):
        self.setup()

        self.feature_file.write(",".join(self.get_headers()) + "\n")
        self.meta_file.write(",".join(self.get_meta_headers()) + "\n")

        features_list=[]
        meta_list=[]
        for packet in tqdm(self.packets):
            meta = self.get_traffic_vector(packet)

            if meta is None:
                self.skipped += 1
                continue

            feature = self.update(meta)
            features_list.append(feature)
            meta_list.append(meta)
            self.count += 1

            if self.count % 1e4 == 0:
                np.savetxt(
                    self.feature_file,
                    np.vstack(features_list),
                    delimiter=",",
                    fmt="%.7f",
                )
                np.savetxt(
                    self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s"
                )
                features_list = []
                meta_list = []

        # save remaining
        np.savetxt(
            self.feature_file, np.vstack(features_list), delimiter=",", fmt="%.7f"
        )
        np.savetxt(self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")

        self.teardown()

        
    
    def get_nstat(self):
        return self.nstat #, self.dummy_nstat

    def update(self, traffic_vector):
        return self.nstat.update_get_stats(*traffic_vector)
        
    def get_traffic_vector(self, packet):
        packet = packet[0]

        # only process IP packets,
        if not (packet.haslayer(IP) or packet.haslayer(IPv6) or packet.haslayer(ARP)):
            return None

        timestamp = packet.time
        framelen = len(packet)
        if packet.haslayer(IP):  # IPv4
            srcIP = packet[IP].src
            dstIP = packet[IP].dst
            IPtype = 0
        elif packet.haslayer(IPv6):  # ipv6
            srcIP = packet[IPv6].src
            dstIP = packet[IPv6].dst
            IPtype = 1
        else:
            srcIP = ''
            dstIP = ''

        if packet.haslayer(TCP):
            srcproto = str(packet[TCP].sport)
            dstproto = str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            srcproto = str(packet[UDP].sport)
            dstproto = str(packet[UDP].dport)
        else:
            srcproto = ''
            dstproto = ''

        if packet.haslayer(ARP):
            srcMAC=packet[ARP].hwsrc
            dstMAC=packet[ARP].hwdst
        else:
            srcMAC = packet.src
            dstMAC = packet.dst

        if srcproto == '':  # it's a L2/L1 level protocol
            if packet.haslayer(ARP):  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = packet[ARP].psrc  # src IP (ARP)
                dstIP = packet[ARP].pdst  # dst IP (ARP)
                IPtype = 0
            elif packet.haslayer(ICMP):  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        traffic_vector=[IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, float(timestamp), int(framelen)]
        
        return traffic_vector
    
    def get_headers(self):
        
        stat_1d= ["weight","mean","std"]

        stat_2d=["magnitude","radius","covariance","pcc"]
        stream_1d=["Src-MAC-IP","Jitter"]
        stream_2d=["Channel","Socket"]
        headers=[]
        for name, stat in itertools.product(stream_1d, stat_1d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        for name, stat in itertools.product(stream_2d, stat_1d+stat_2d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        
        return headers
    
    def get_meta_headers(self):
        return ["ip_type", "src_mac", "dst_mac", "scr_ip", "src_protocol", "dst_ip", "dst_protocol", "time_stamp", "packet_size"]

def magnitude(mean, other_mean):
    # the magnitude of a set of incStats, pass var instead of mean to get radius
    return np.sqrt(np.power(mean,2)+np.power(other_mean,2))

class IncStat1D:
    """
    Incremental Statistics for 1 dimensional features stores a table of size 3*decay_factors
    """    
    def __init__(self, decay_factors, name, t, v=None):  # timestamp is creation time
        self.name = name
        self.last_timestamp = t
        if v is None:
            v = t - self.last_timestamp
        self.incremental_statistics=np.tile( np.expand_dims([1., v, v**2], axis=1), [1,len(decay_factors)]) # each row corresponds to weight, linear sum, sum of squares
        self.weight_thresh=1e-6
        self.decay_factors = np.array(decay_factors)  # Decay Factor


    def __repr__(self):
        return pformat(vars(self))

    def insert(self, t, v=None):  
        
        # special case for jitter
        if v is None:
            v = t - self.last_timestamp

        self.decay(t)

        # update with v
        self.incremental_statistics += np.expand_dims([1., v, v**2], axis=1) #broadcast to [3,1]
        self.last_timestamp = t

    def decay(self, t):
        # check for decay
        time_diff=t - self.last_timestamp
        if time_diff >= 0:
            factor = 2. ** (-self.decay_factors * time_diff)
            self.incremental_statistics *= factor
            
    def is_outdated(self, t):
        time_diff=t - self.last_timestamp
        if time_diff >= 0:
            factor = 2 ** (-self.decay_factors * time_diff)
        else:
            print(t)
            print(self)
            print(time_diff)
        
        if np.all((self.incremental_statistics[0]*factor)<self.weight_thresh):
            return True 
        else:
            return False

    def weight(self):
        return self.incremental_statistics[0]

    def mean(self):
        return np.where(self.weight()<self.weight_thresh, 0., self.incremental_statistics[1] / self.incremental_statistics[0])

    def var(self):
        return np.where(self.weight()<self.weight_thresh, 0., np.abs(self.incremental_statistics[2] / self.weight() - self.mean()**2))
    
    def std(self):
        return np.sqrt(self.var())

    def all_stats_1D(self):
        return np.hstack([self.weight(),self.mean(), self.std()])

class IncStat2D:
    def __init__(self, incS1, incS2, t):
        # store references to the streams' incStats
        self.inc_stats1=incS1
        self.inc_stats2=incS2
        self.decay_factors=incS1.decay_factors
        self.sum_of_residual = np.zeros(len(self.decay_factors)) # sum of residual products (A-uA)(B-uB)
        self.eps=1e-6
        self.last_timestamp=t
        
    def __repr__(self):
        return "{}, {} : ".format(self.inc_stats1.name,self.inc_stats2.name)+pformat(self.all_stats_2D())

    def update_cov(self, t, v):
        # Decay residules
        self.decay(t)

        # Compute and update residule
        r_ij = (v-self.inc_stats1.mean())*(v-self.inc_stats2.mean())
        self.sum_of_residual += r_ij
        self.last_timestamp=t

    def decay(self, t):
        # check for decay cf3
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2 ** (-self.decay_factors * time_diff)
            self.sum_of_residual *= factor           

    #covariance approximation
    def cov(self):
        weight_sum=self.inc_stats1.weight()+self.inc_stats2.weight()
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.where(weight_sum<self.eps, 0., self.sum_of_residual / weight_sum) 
        # return self.sr / self.w3

    # Pearson corl. coef
    def pcc(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            ss = self.inc_stats1.std() * self.inc_stats2.std()
            return np.where(ss<self.eps, 0., self.cov() / ss)
            
            
    def all_stats_2D(self):
        mean1=self.inc_stats1.mean()
        mean2=self.inc_stats2.mean()
        var1=self.inc_stats1.var()
        var2=self.inc_stats2.var()
        return np.hstack([magnitude(mean1,mean2),magnitude(var1,var2), self.cov(), self.pcc()])

class IncStatDB:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self,  decay_factors, limit=1e5):
        # list of dictionary to store 1d stats for each lambda, index matches lambda id
        self.stat1d = {}
        # list of dict to store 2d stats for each lambda, index matches lambda id
        self.stat2d = {}
        # limit for all lambdas combined
        self.limit = limit
        self.num_entries=0
        self.decay_factors=decay_factors
        
        # number of pkts updated
        self.num_updated=0

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    def update_get_stats_1D(self, ID, t, v):
        # not in our db
        if ID not in self.stat1d or self.stat1d[ID] is None:
            if self.num_entries + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + ID + '\nwould exceed incStat 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
                
            self.stat1d[ID] = IncStat1D(self.decay_factors, ID, t, v)

            self.num_entries+=1
        else:
            self.stat1d[ID].insert(t, v)
        
        return self.stat1d[ID].all_stats_1D()    
        
    
    # Registers covariance tracking for two streams, registers missing streams
    def update_get_stats_2D(self, ID1, ID2, t, v):

        #update 1d after covariance
        stats1d=self.update_get_stats_1D(ID1, t, v)
        
        # the destination receives negative value
        self.update_get_stats_1D(ID2, t, -v)
        
        #check for pre-exiting link
        if frozenset([ID1, ID2]) not in self.stat2d or self.stat2d[frozenset([ID1, ID2])] is None:
            # Link incStats
            inc_cov = IncStat2D(self.stat1d[ID1],self.stat1d[ID2], t)
            self.stat2d[frozenset([ID1, ID2])]=inc_cov
            
        self.stat2d[frozenset([ID1, ID2])].update_cov(t, v)
        stats2d=self.stat2d[frozenset([ID1, ID2])].all_stats_2D()
        
        return np.hstack([stats1d,stats2d])

    #cleans out records that have a weight less than the cutoff.
    #returns number of removed records and records looked through
    def clean_records(self, t):
        removed = 0
        for key, inc_stat in dict(self.stat1d).items():
            if inc_stat.is_outdated(t):
                # remove 2d links
                for cov_key in dict(self.stat2d).keys():
                    if key in cov_key:
                        del self.stat2d[cov_key]
                # remove self
                del self.stat1d[key]
                removed+=1
        return removed
     
class NetStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, decay_factors = [5,3,1,0.1,0.01], limit=1e5, log_path=None):
        self.decay_factors=decay_factors
        
        self.clean_up_round=5000
        
        #streams (No need for SrcIP stream because it is already included in Channel statistics)
        self.inc_stat_db = IncStatDB(decay_factors, limit=limit) # from this packet’s source MAC and IP


        print("log_path:", log_path)
        if log_path is not None:
            print("netstat log_path",log_path)
            self.log_file=open(log_path,"w")
        else:
            self.log_file = None

    def set_netstat_log_path(self, log_path):
        print("netstat log_path",log_path)
        self.log_file = open(log_path,"w")

    def get_db(self):
        return {"inc_stat":self.inc_stat_db,
        "num_updated":self.num_updated}

    def set_db(self, db_dict):
        self.inc_stat_db=db_dict["inc_stat"]
        self.num_updated=db_dict["num_updated"]

    def __repr__(self):
        return pformat(self.inc_stat_db, indent=2)

    def get_records(self,IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        """gets deep copy of all relevant records from inc stat db

        Args:
            IPtype (_type_): _description_
            srcMAC (_type_): _description_
            dstMAC (_type_): _description_
            srcIP (_type_): _description_
            srcProtocol (_type_): _description_
            dstIP (_type_): _description_
            dstProtocol (_type_): _description_
            timestamp (_type_): _description_
            datagramSize (_type_): _description_

        Returns:
            _type_: _description_
        """        
        dummy_db=IncStatDB(self.decay_factors)
        
        
        dummy_db.stat1d[f"{srcMAC}_{srcIP}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{srcMAC}_{srcIP}")) # srcMAC-IP
        dummy_db.stat1d[f"{srcIP}_{dstIP}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{srcIP}_{dstIP}")) #jitter
        
        # channel
        dummy_db.stat1d[f"{srcIP}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{srcIP}"))
        dummy_db.stat1d[f"{dstIP}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{dstIP}"))
        dummy_db.stat2d[frozenset([f"{dstIP}",f"{srcIP}"])]=copy.deepcopy(self.inc_stat_db.stat2d.get(frozenset([f"{dstIP}",f"{srcIP}"])))
        
        #socket
        if srcProtocol=='arp':
            dummy_db.stat1d[f"{srcMAC}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{srcMAC}"))
            dummy_db.stat1d[f"{dstMAC}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{dstMAC}"))
            dummy_db.stat2d[frozenset([f"{srcMAC}",f"{dstMAC}"])]=copy.deepcopy(self.inc_stat_db.stat2d.get(frozenset([f"{dstMAC}",f"{srcMAC}"])))
        else:
            dummy_db.stat1d[f"{srcIP}_{srcProtocol}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{srcIP}_{srcProtocol}"))
            dummy_db.stat1d[f"{dstIP}_{dstProtocol}"]=copy.deepcopy(self.inc_stat_db.stat1d.get(f"{dstIP}_{dstProtocol}"))
            dummy_db.stat2d[frozenset([f"{srcIP}_{srcProtocol}",f"{dstIP}_{dstProtocol}"])]=copy.deepcopy(self.inc_stat_db.stat2d.get(frozenset([f"{srcIP}_{srcProtocol}",f"{dstIP}_{dstProtocol}"])))

        dummy_db.num_updated=self.inc_stat_db.num_updated
        dummy_db.num_entries=self.inc_stat_db.num_entries
        return dummy_db

    def update_get_stats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize, db=None):
        if db is None:
            db=self.inc_stat_db
        
        
        if self.log_file is not None:
            self.log_file.write(f"{IPtype}, {srcMAC}, {dstMAC}, {srcIP}, {srcProtocol}, {dstIP}, {dstProtocol}, {timestamp}, {datagramSize}\n")

        #streams (No need for SrcIP stream because it is already included in Channel statistics)
        src_mac_ip=db.update_get_stats_1D(f"{srcMAC}_{srcIP}", timestamp, datagramSize) # srcMAC-IP from this packet’s source MAC and IP
        jitter=db.update_get_stats_1D(f"{srcIP}_{dstIP}", timestamp, None) # jitter between channels
        channel=db.update_get_stats_2D(f"{srcIP}", f"{dstIP}", timestamp, datagramSize) # channel: sent between this packet’s source and destination IPs
        
        # Socket: sent between this packet’s source and destination TCP/UDP Socket
        # arp has no IP 
        if srcProtocol=='arp':
            socket=db.update_get_stats_2D(f"{srcMAC}", f"{dstMAC}", timestamp, datagramSize) 
        else:
            socket=db.update_get_stats_2D(f"{srcIP}_{srcProtocol}", f"{dstIP}_{dstProtocol}", timestamp, datagramSize)
            
        db.num_updated+=1

        #clean our records every 100 updates
        if db.num_updated%self.clean_up_round==0:
            db.clean_records(timestamp)
            
        return np.hstack([src_mac_ip, jitter, channel, socket])
    
