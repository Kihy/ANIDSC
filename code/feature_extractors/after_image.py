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


class AfterImage(BaseTrafficFeatureExtractor):
    def __init__(self, limit=float("inf"), decay_factors=[5, 3, 1, 0.1, 0.01], max_pkt=float("inf"), **kwargs):
        """initializes afterimage, a packet-based feature extractor used in Kitsune

        Args:
            limit (int, optional): maximum number of records. Defaults to 1e6.
            decay_factors (list, optional): the time windows. Defaults to [5,3,1,.1,.01].
        """
        super().__init__(**kwargs)
        self.limit = limit
        self.decay_factors = decay_factors
        self.name = "AfterImage"
        self.max_pkt =max_pkt

    def setup(self):
        """sets up after image"""
        super().setup()
        if self.reset_state:
            self.state = NetStat(
                decay_factors=self.decay_factors,
                limit=self.limit,
            )

    def peek(self, traffic_vectors):
        """fake update. obtains a copy of existing database,
        applies the traffic vectors to it.

        Args:
            traffic_vectors (2d array): list of traffic vectors to be updated

        Returns:
            2d array: the corresponding features
        """

        fake_db = self.state.get_records(traffic_vectors)
        vectors = []
        for tv in traffic_vectors:
            vectors.append(self.state.update_get_stats(*tv, fake_db))
        return vectors

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
        for packet in tqdm(self.input_pcap, desc=f"parsing {self.file_name}"):
            if self.count>self.max_pkt:
                break
            
            traffic_vector = self.get_traffic_vector(packet)
            
            if traffic_vector is None:
                self.skipped += 1
                continue

            if self.offset_time is None and self.offset_timestamp:
                self.offset_time = traffic_vector[-2] - self.state.last_timestamp
            else:
                self.offset_time = 0
            traffic_vector[-2] -= self.offset_time



            feature = self.update(traffic_vector)
            features_list.append(feature)
            meta_list.append(traffic_vector)
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

    def update(self, traffic_vector):
        """updates the internal state with traffic vector

        Args:
            traffic_vector (array): a traffic vectors consists of
            data extracted from pacekts

        Returns:
            array: the extracted features
        """
        return self.state.update_get_stats(*traffic_vector)

    def get_traffic_vector(self, packet):
        """extracts the traffic vectors from packet

        Args:
            packet (scapy packet): input packet

        Returns:
            array: list of IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, time, packet size
        """
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
                IPtype = 0
            elif packet.haslayer(ICMP):  # is ICMP
                srcproto = "icmp"
                dstproto = "icmp"
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == "":  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        traffic_vector = [
            IPtype,
            srcMAC,
            dstMAC,
            srcIP,
            srcproto,
            dstIP,
            dstproto,
            float(timestamp),
            int(framelen),
        ]

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
        for name, stat in itertools.product(stream_1d, stat_1d):
            for time in self.decay_factors:
                headers.append(f"{name}_{time}_{stat}")
        for name, stat in itertools.product(stream_2d, stat_1d + stat_2d):
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


def magnitude(x, y):
    """the magnitude of a set of incStats, pass var instead of mean to get radius

    Args:
        x (float): first value
        y (float): second value

    Returns:
        float: result
    """
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


class IncStat1D:
    def __init__(self, decay_factors, name, t, v=None):  # timestamp is creation time
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
        self.name = name
        self.last_timestamp = t
        if v is None:
            v = t - self.last_timestamp
        self.incremental_statistics = np.tile(
            np.expand_dims([1.0, v, v**2], axis=1), [1, len(decay_factors)]
        )  # each row corresponds to weight, linear sum, sum of squares
        self.weight_thresh = 1e-3
        self.decay_factors = np.array(decay_factors)  # Decay Factor

    def __repr__(self):
        return pformat(vars(self))

    def insert(self, t, v=None):
        """updates the incstat with value.
        decays the statistics and increments it.

        Args:
            t (float): time of new information
            v (float, optional): value of new information. use None to measure
            jitter. Defaults to None.
        """
        # special case for jitter
        if v is None:
            v = t - self.last_timestamp

        self.decay(t)

        # update with v
        self.incremental_statistics += np.expand_dims(
            [1.0, v, v**2], axis=1
        )  # broadcast to [3,1]
        self.last_timestamp = t

    def decay(self, t):
        """decays the incremental statistics according to t

        Args:
            t (float): current timestamp
        """
        # check for decay
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2.0 ** (-self.decay_factors * time_diff)
            self.incremental_statistics *= factor

    def is_outdated(self, t):
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
            factor = 2 ** (-self.decay_factors * time_diff)
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
    def __init__(self, incS1, incS2, t):
        """2 dimensional IncStat, stores the relationship between two
        1d inc stat. keeps track of the sum of residual products (A-uA)(B-uB)

        Args:
            incS1 (IncStat): first incstat
            incS2 (IncStat): second incstat
            t (float): current timestamp
        """
        # store references to the streams' incStats
        self.inc_stats1 = incS1
        self.inc_stats2 = incS2
        self.decay_factors = incS1.decay_factors
        self.sum_of_residual = np.zeros(len(self.decay_factors))
        self.eps = 2e-3
        self.last_timestamp = t

    def __repr__(self):
        return "{}, {} : ".format(self.inc_stats1.name, self.inc_stats2.name) + pformat(
            self.all_stats_2D()
        )

    def update_cov(self, t, v):
        """updates the covariance of the two streams.
        decays the residual then updates it.

        Args:
            t (float): current time
            v (float): the value to be updated
        """
        # Decay residules
        self.decay(t)

        # Compute and update residule
        self.sum_of_residual += (v - self.inc_stats1.mean()) * (
            v - self.inc_stats2.mean()
        )
        self.last_timestamp = t

    def decay(self, t):
        """decays the residual product. ignores if time diff is negative

        Args:
            t (float): current timestamp
        """
        time_diff = t - self.last_timestamp
        if time_diff >= 0:
            factor = 2 ** (-self.decay_factors * time_diff)
            self.sum_of_residual *= factor

    def cov(self):
        """calculates the covariance between two incstats:
        sum of redsidual/(w1+w2). if weight sum is too small, it will return 0

        Returns:
            array: covariance of each time window
        """
        weight_sum = self.inc_stats1.weight() + self.inc_stats2.weight()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                weight_sum < self.eps, 0.0, self.sum_of_residual / weight_sum
            )

    def pcc(self):
        """pearson correlation coefficient. calculated via cov/(std1*std2)

        Returns:
            float: pcc of each timestamp
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ss = self.inc_stats1.std() * self.inc_stats2.std()
            return np.where(ss < self.eps, 0.0, self.cov() / ss)

    def all_stats_2D(self):
        """calculates all 2d statistics

        Returns:
            array: [magnitude, radius, cov, pcc]
        """
        mean1 = self.inc_stats1.mean()
        mean2 = self.inc_stats2.mean()
        var1 = self.inc_stats1.var()
        var2 = self.inc_stats2.var()
        return np.hstack(
            [magnitude(mean1, mean2), magnitude(var1, var2), self.cov(), self.pcc()]
        )


class IncStatDB:
    def __init__(self, decay_factors, limit=1e5):
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
        self.decay_factors = decay_factors

        # number of pkts updated
        self.num_updated = 0

    def update_get_stats_1D(self, ID, t, v):
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
        if ID not in self.stat1d or self.stat1d[ID] is None:
            if self.num_entries + 1 > self.limit:
                raise LookupError(
                    "Adding Entry:\n"
                    + ID
                    + "\nwould exceed incStat 1D limit of "
                    + str(self.limit)
                    + ".\nObservation Rejected."
                )

            self.stat1d[ID] = IncStat1D(self.decay_factors, ID, t, v)

            self.num_entries += 1
        else:
            self.stat1d[ID].insert(t, v)

        return self.stat1d[ID].all_stats_1D()

    def update_get_link_stats(self, ID1, ID2, t, v):
        """create 2d incstat if it doesnt exist and updates it.
        returns only the 2d statistics

        Args:
            ID1 (str): ID of incstat1
            ID2 (str): ID of incstat2
            t (float): current timestamp
            v (float): value

        Returns:
            array: 1d+2d statistics
        """
        
        # update 1d after covariance
        self.update_get_stats_1D(ID1, t, v)

        # the destination receives negative value
        self.update_get_stats_1D(ID2, t, -v)

        # check for pre-exiting link
        if (
            (ID1, ID2) not in self.stat2d
            or self.stat2d[(ID1, ID2)] is None
        ):
            # Link incStats
            inc_cov = IncStat2D(self.stat1d[ID1], self.stat1d[ID2], t)
            self.stat2d[(ID1, ID2)] = inc_cov

        self.stat2d[(ID1, ID2)].update_cov(t, v)
        stats2d = self.stat2d[(ID1, ID2)].all_stats_2D()

        return stats2d
    
    def update_get_stats_2D(self, ID1, ID2, t, v):
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

        # update 1d after covariance
        stats1d = self.update_get_stats_1D(ID1, t, v)

        # the destination receives negative value
        self.update_get_stats_1D(ID2, t, -v)

        # check for pre-exiting link
        if (
            frozenset([ID1, ID2]) not in self.stat2d
            or self.stat2d[frozenset([ID1, ID2])] is None
        ):
            # Link incStats
            inc_cov = IncStat2D(self.stat1d[ID1], self.stat1d[ID2], t)
            self.stat2d[frozenset([ID1, ID2])] = inc_cov

        self.stat2d[frozenset([ID1, ID2])].update_cov(t, v)
        stats2d = self.stat2d[frozenset([ID1, ID2])].all_stats_2D()

        return np.hstack([stats1d, stats2d])

    def clean_records(self, t):
        """cleans out records that have small weight

        Args:
            t (float): current timestamp

        Returns:
            int: number of removed records and records looked through
        """
        removed = 0
        removed_keys=[]
        for key, inc_stat in dict(self.stat1d).items():
            if inc_stat.is_outdated(t):
                # remove 2d links
                for cov_key in dict(self.stat2d).keys():
                    if key in cov_key:
                        del self.stat2d[cov_key]
                        removed_keys.append(cov_key)
                # remove self
                del self.stat1d[key]
                removed += 1
        return removed, removed_keys


class NetStat:
    def __init__(self, decay_factors=[5, 3, 1, 0.1, 0.01], limit=1e5):
        """Datastructure for efficent network stat queries

        Args:
            decay_factors (list, optional): time windows. Defaults to [5,3,1,0.1,0.01].
            limit (int, optional): maximum number of incstats. Defaults to 1e5.
        """
        self.decay_factors = decay_factors
        self.last_timestamp = None
        self.clean_up_round = 5000

        # streams (No need for SrcIP stream because it is already included in Channel statistics)
        self.inc_stat_db = IncStatDB(
            decay_factors, limit=limit
        )  # from this packet’s source MAC and IP
        

    def __repr__(self):
        return pformat(self.inc_stat_db, indent=2)

    def get_records(self, traffic_vectors):
        """get records associated with all traffic vectors

        Args:
            traffic_vectors (list): list of all traffic vectors

        Returns:
            IncstatDB: database containing relevant records
        """
        dummy_db = IncStatDB(self.decay_factors)

        for (
            IPtype,
            srcMAC,
            dstMAC,
            srcIP,
            srcProtocol,
            dstIP,
            dstProtocol,
            timestamp,
            datagramSize,
        ) in traffic_vectors:

            dummy_db.stat1d[f"{srcMAC}_{srcIP}"] = copy.deepcopy(
                self.inc_stat_db.stat1d.get(f"{srcMAC}_{srcIP}")
            )  # srcMAC-IP
            dummy_db.stat1d[f"{srcIP}_{dstIP}"] = copy.deepcopy(
                self.inc_stat_db.stat1d.get(f"{srcIP}_{dstIP}")
            )  # jitter

            # channel
            dummy_db.stat1d[f"{srcIP}"] = copy.deepcopy(
                self.inc_stat_db.stat1d.get(f"{srcIP}")
            )
            dummy_db.stat1d[f"{dstIP}"] = copy.deepcopy(
                self.inc_stat_db.stat1d.get(f"{dstIP}")
            )
            dummy_db.stat2d[frozenset([f"{dstIP}", f"{srcIP}"])] = copy.deepcopy(
                self.inc_stat_db.stat2d.get(frozenset([f"{dstIP}", f"{srcIP}"]))
            )

            # socket
            if srcProtocol == "arp":
                dummy_db.stat1d[f"{srcMAC}"] = copy.deepcopy(
                    self.inc_stat_db.stat1d.get(f"{srcMAC}")
                )
                dummy_db.stat1d[f"{dstMAC}"] = copy.deepcopy(
                    self.inc_stat_db.stat1d.get(f"{dstMAC}")
                )
                dummy_db.stat2d[frozenset([f"{srcMAC}", f"{dstMAC}"])] = copy.deepcopy(
                    self.inc_stat_db.stat2d.get(frozenset([f"{dstMAC}", f"{srcMAC}"]))
                )
            else:
                dummy_db.stat1d[f"{srcIP}_{srcProtocol}"] = copy.deepcopy(
                    self.inc_stat_db.stat1d.get(f"{srcIP}_{srcProtocol}")
                )
                dummy_db.stat1d[f"{dstIP}_{dstProtocol}"] = copy.deepcopy(
                    self.inc_stat_db.stat1d.get(f"{dstIP}_{dstProtocol}")
                )
                dummy_db.stat2d[
                    frozenset([f"{srcIP}_{srcProtocol}", f"{dstIP}_{dstProtocol}"])
                ] = copy.deepcopy(
                    self.inc_stat_db.stat2d.get(
                        frozenset([f"{srcIP}_{srcProtocol}", f"{dstIP}_{dstProtocol}"])
                    )
                )

        dummy_db.num_updated = self.inc_stat_db.num_updated
        dummy_db.num_entries = self.inc_stat_db.num_entries
        return dummy_db

    def update_get_stats(
        self,
        IPtype,
        srcMAC,
        dstMAC,
        srcIP,
        srcProtocol,
        dstIP,
        dstProtocol,
        timestamp,
        datagramSize,
        db=None,
    ):
        """updates the netstat with traffic vectors

        Args:
            IPtype (int): IP type, 0 for IPv4, and 1 for IPv6, not really used
            srcMAC (str): source MAC
            dstMAC (str): destination MAC
            srcIP (str): source IP
            srcProtocol (int): source port
            dstIP (str): destination IP
            dstProtocol (int): destination port
            timestamp (float): arrival time
            datagramSize (float): packet size
            db (IncStatDB, optional): dummy database, if None, updates real one. Defaults to None.

        Returns:
            array: features extracted
        """
        if db is None:
            db = self.inc_stat_db

        self.last_timestamp = timestamp

        # streams (No need for SrcIP stream because it is already included in Channel statistics)
        
        # srcMAC-IP from this packet’s source MAC and IP
        src_mac_ip = db.update_get_stats_1D(
            f"{srcMAC}_{srcIP}", timestamp, datagramSize
        )  
        
         # jitter between channels
        jitter = db.update_get_stats_1D(
            f"{srcIP}_{dstIP}", timestamp, None
        )  
        
        # channel: sent between this packet’s source and destination IPs
        channel = db.update_get_stats_2D(
            f"{srcIP}", f"{dstIP}", timestamp, datagramSize
        )  
        

        # Socket: sent between this packet’s source and destination TCP/UDP Socket
        # arp has no IP
        if srcProtocol == "arp":
            socket = db.update_get_stats_2D(
                f"{srcMAC}", f"{dstMAC}", timestamp, datagramSize
            )
        else:
            socket = db.update_get_stats_2D(
                f"{srcIP}_{srcProtocol}",
                f"{dstIP}_{dstProtocol}",
                timestamp,
                datagramSize,
            )

        db.num_updated += 1

        # clean our records
        if db.num_updated % self.clean_up_round == 0:
            db.clean_records(timestamp)

        return np.hstack([src_mac_ip, jitter, channel, socket])
