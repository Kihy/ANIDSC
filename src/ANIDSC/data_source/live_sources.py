from scapy.all import sniff
from ..base_files import PipelineSource

class LiveSniffer(PipelineSource):
    def __init__(self, iface:str):
        self.iface=iface 
        self.context={}
    def start(self):
        try:
            sniff(iface=self.iface, prn=self.call_back, count=0)
        except KeyboardInterrupt as e:
            print("Interrupted")
        self.on_end()
