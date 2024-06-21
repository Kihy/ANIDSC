from scapy.all import sniff
from ..base_files import PipelineSource

class LiveSniffer(PipelineSource):
    def __init__(self, iface:str, max_pkts:int=0):
        """live capture from iface

        Args:
            iface (str): name of interface
            max_pkts(int): maximum number of packets to capture set to 0 to capture forever. defaults to 0.
        """        
        self.iface=iface
        self.max_pkts=max_pkts 
        self.context={"iface":iface}


    def start(self):
        """start the live sniffer
        """        
        try:
            sniff(iface=self.iface, prn=self.call_back, count=self.max_pkts)
        except KeyboardInterrupt as e:
            print("Interrupted")
        self.on_end()
