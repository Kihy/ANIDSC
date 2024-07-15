import copy
import numpy as np
import pyswarms as ps
from scapy.all import Packet, TCP, UDP, ARP, Raw, IP
from ..base_files.pipeline import PipelineComponent

class LiuerMihouAttack(PipelineComponent):
    def __init__(
        self,
        feature_extractor,
        model,        
        upper_bounds=[0.1,5,1514],
        max_modified=float("inf"),
        **kwargs        
    ):
        super().__init__(component_type="adversarial_attacks", **kwargs)
        self.feature_extractor=feature_extractor
        self.model=model
        self.upper_bounds = upper_bounds
        self.pso_config = {
            "n_particles": 30,
            "iterations": 20,
            "options": {"c1": 0.7, "c2": 0.3, "w": 0.5},
            "p": 2,
            "k": 4,
            "clamp": None,
        }
        self.max_modified=max_modified
        self.processed=0
        self.modified=0
        
    def setup(self):
        self.parent.context["adversarial_attack"]=self.name

    def process(self, packet:Packet):
        
        # skip processing if max is reached        
        if self.modified > self.max_modified:
            self.processed+=1
            return packet

        feature, traffic_vector = self.feature_extractor.process(packet, peek=True)
        
        self.processed+=1
        
        if traffic_vector == None:
            return packet 
        
        score, threshold=self.model.predict_step(feature)
        
        if score<threshold:  # packet is benign
            return packet

        min_payload = len(packet) - len(packet.payload)

        self.pso_config['bounds'] = np.array(
            [
                [0., 0.0, min_payload],
                self.upper_bounds
            ]
        )
        self.pso_config['dimensions']=3

        cost, pos = self.optimize(packet)

        self.modified+=1 
        
        delay, n_craft, payload = pos[0], int(pos[1]), int(pos[2])

        adversarial_packets=self.particle_to_packet(packet, delay, n_craft, payload)

        return adversarial_packets

    def particle_to_packet(self, packet, delay, n_craft, payload):
        adversarial_packets=[]
        
        # generate fake packets
        delay_times = np.linspace(0, delay, n_craft + 1, endpoint=False)[1:]
        for i in range(n_craft):
            change_in_payload=payload-len(packet) 
            
            craft_packet = self.packet_gen(
                packet, delay_times[i], change_in_payload
            )
            adversarial_packets.append(craft_packet)

        adversarial_packet = self.packet_gen(
            packet, delay, 0
        )
        adversarial_packets.append(adversarial_packet)
        
        return adversarial_packets
            
    def packet_gen(self, packet, time, size):
        new_packet=copy.deepcopy(packet)
        
        # delay arrival time
        new_packet.time += time

        if size !=0:
            if packet.haslayer(TCP):
                new_packet[TCP].remove_payload()
                payload_size = len(packet) + size
                new_packet[TCP].add_payload(Raw(load="a" * payload_size))
            del packet[TCP].chksum

            if packet.haslayer(UDP):
                new_packet[UDP].remove_payload()
                payload_size = len(packet) + size
                new_packet[UDP].add_payload(Raw(load="a" * payload_size))

        if packet.haslayer(IP):
            del packet[IP].len
            del packet[IP].chksum
            del packet.len

        return packet

    def optimize(self, packet):
        optimizer = ps.single.GlobalBestPSO(**self.pso_config)

        cost, pos = optimizer.optimize(self.cost_function_of(packet), iters=1000)

        return cost, pos

    def cost_function_of(packet):
        def cost_function(self, x):        
            cost = []
            # iterative through each particle
            for time_delay, n_craft, packet_size in x:
                packets=self.particle_to_packet(packet, time_delay, n_craft, packet_size)
                
                features=[self.feature_extractor.process(p, peek=True) for p in packets]
                
                anomaly_scores, threshold = self.model.predict_step(np.vstack(features))
                
                cost.append(np.max(anomaly_scores-threshold))

            return np.array(cost)
        return cost_function

    