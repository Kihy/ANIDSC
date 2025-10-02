from socket import getservbyport

from ..feature_buffer.tabular import TabularFeatureBuffer
from ..component.feature_extractor import BaseMetaExtractor
from ..save_mixin.pickle import PickleSaveMixin


from scapy.all import IP, IPv6, TCP, UDP, ARP, ICMP, Packet

class ProtocolMetaExtractor(PickleSaveMixin, BaseMetaExtractor):
    def __init__(self,  protocol_map={},**kwargs):
        super().__init__(**kwargs)
        self.protocol_map=protocol_map

        
    def setup(self):
        pass 
    
    def get_meta_vector(self, packet: Packet):
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

        if TCP in packet:
            protocol = self.get_protocol_name(packet.sport, packet.dport, "tcp")
        elif UDP in packet:
            protocol = self.get_protocol_name(packet.sport, packet.dport, "udp")
        elif ARP in packet:
            protocol = "ARP"
        elif ICMP in packet:
            protocol = "ICMP"
        else:
            protocol = "Other"

        traffic_vector = {
            "protocol": protocol,
            "srcMAC": srcMAC,
            "dstMAC": dstMAC,
            "srcIP": srcIP,
            "srcport": srcproto,
            "dstIP": dstIP,
            "dstport": dstproto,
            "timestamp": float(timestamp),
            "packet_size": int(framelen),
        }

        return traffic_vector

    def get_protocol_name(
        self, src_port_num: int, dst_port_num: int, proto: str
    ) -> str:
        """gets the protocol name associated with port number

        Args:
            src_port_num (int): source port number
            dst_port_num (int): destination port number
            proto (str): protocol, either 'udp' or 'tcp'

        Returns:
            str: protocol string for the port
        """
        try:
            protocol = getservbyport(src_port_num, proto)
        except OSError:
            try:
                protocol = getservbyport(dst_port_num, proto)
            except OSError:
                protocol = proto

        if protocol not in self.protocol_map.keys():
            protocol = proto

        return protocol.upper()

    @property
    def headers(self):
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
