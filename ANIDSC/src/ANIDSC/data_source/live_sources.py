from scapy.all import sniff
from ..component.pipeline_source import PipelineSource

class LiveSniffer(PipelineSource):
    def __init__(self, iface:str, **kwargs):
        """live capture from iface

        Args:
            iface (str): name of interface
            max_pkts(int): maximum number of packets to capture set to 0 to capture forever. defaults to 0.
        """        
        super().__init__(**kwargs)
        self.iface=iface
        self.context={"iface":iface}


    def start(self):
        """start the live sniffer
        """
        self.on_start()
        try:
            sniff(iface=self.iface, prn=self.other.process, count=self.max_records)
        except KeyboardInterrupt as e:
            print("Interrupted")
        self.on_end()
