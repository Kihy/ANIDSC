import copy
from pathlib import Path
import numpy as np
import pyswarms as ps
from scapy.all import Packet, TCP, UDP, ARP, Raw, IP, PcapWriter
from ..base_files.pipeline import PipelineComponent


class PcapSaver(PipelineComponent):
    def __init__(self, **kwargs):
        super().__init__(component_type="", **kwargs)

    def setup(self):
        context = self.get_context()
        # setup files
        self.pcap_path = Path(
            f"{context['dataset_name']}/pcap/{context['file_name']}.pcap"
        )
        self.pcap_path.parent.mkdir(parents=True, exist_ok=True)

        self.pcap_file = PcapWriter(str(self.pcap_path))

    def process(self, data):
        if isinstance(data, list):
            for i in data:
                self.pcap_file.write(i)
        else:
            self.pcap_file.write(data)
        return data

    def teardown(self):
        print(f"pcap saved to {self.pcap_path}")
        self.pcap_file.close()

    def __str__(self):
        return f"PcapSaver"


class LiuerMihouAttack(PipelineComponent):
    def __init__(
        self,
        feature_extractor,
        model,
        standardizer=None,
        upper_bounds=[0.1, 5, 1514],
        max_modified=float("inf"),
        **kwargs,
    ):
        super().__init__(component_type="", **kwargs)
        self.feature_extractor = feature_extractor
        self.model = model
        self.standardizer = standardizer
        self.upper_bounds = upper_bounds
        self.pso_config = {
            "n_particles": 30,
            "dimensions": 3,
            "options": {"c1": 0.7, "c2": 0.3, "w": 0.5, "p": 2, "k": 4},
        }
        self.iters = 20
        self.max_modified = max_modified
        self.processed = 0
        self.modified = 0

    def setup(self):
        super().setup()
        context=self.get_context()
        self.parent.context["file_name"] = (
            f"{self.name}/{self.parent.context['file_name']}"
        )
    
        log_path = Path(
            f"{context['dataset_name']}/{self.feature_extractor.name}/attack_results/{self.name}/{context['file_name']}/{context['pipeline_name']}.csv"
        )
    
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(str(log_path), "w")
        header=["idx", "ori_cost", "new_cost", "delay","n_craft","payload"]
        self.log_file.write(",".join(header) + "\n")

    def process(self, packet: Packet):
        # skip processing if max is reached
        if self.modified > self.max_modified:
            self.processed += 1
            return packet

        feature, traffic_vector = self.feature_extractor.process(packet, peek=True)

        self.processed += 1

        if traffic_vector is None:
            return packet

        anomaly_scores = self.calc_anomaly_scores(feature)

        if anomaly_scores < 0:  # packet is benign
            return packet

        min_payload = len(packet) - len(packet.payload)

        self.pso_config["bounds"] = np.array(
            [[0.0, 0.0, min_payload], self.upper_bounds]
        )

        cost, pos = self.optimize(packet)
        
        self.modified += 1

        delay, n_craft, payload = pos[0], int(pos[1]), int(pos[2])

        adversarial_packets = self.particle_to_packet(packet, delay, n_craft, payload)

        values=[str(self.processed), str(anomaly_scores), str(cost), f"{delay},{n_craft},{payload}"]
        self.log_file.write(",".join(values) + "\n")

        return adversarial_packets

    def particle_to_packet(self, packet, delay, n_craft, payload):
        adversarial_packets = []

        # round values to nearest integer
        n_craft = np.rint(n_craft).astype(int)
        payload = np.rint(payload).astype(int)

        # generate fake packets
        delay_times = np.linspace(0, delay, n_craft + 1, endpoint=False)[1:]
        for t in delay_times:
            change_in_payload = payload - len(packet)

            craft_packet = self.packet_gen(packet, t, change_in_payload)
            adversarial_packets.append(craft_packet)

        adversarial_packet = self.packet_gen(packet, delay, 0)
        adversarial_packets.append(adversarial_packet)

        return adversarial_packets

    def packet_gen(self, packet, time, size):
        new_packet = copy.deepcopy(packet)

        # delay arrival time
        new_packet.time += time

        if size != 0:
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

        cost, pos = optimizer.optimize(self.cost_function_of(packet), iters=self.iters)

        return cost, pos

    def calc_anomaly_scores(self, features):
        if self.standardizer is not None:
            features = self.standardizer.process(features)

        anomaly_scores, threshold = self.model.predict_step(features, preprocess=True)

        return np.max(anomaly_scores - threshold)

    def cost_function_of(self, packet):
        def cost_function(x):
            cost = []
            # iterative through each particle
            for time_delay, n_craft, packet_size in x:
                packets = self.particle_to_packet(
                    packet, time_delay, n_craft, packet_size
                )

                # get only feature
                features = [
                    self.feature_extractor.process(p, peek=True)[0] for p in packets
                ]

                cost.append(self.calc_anomaly_scores(np.vstack(features)))

            return np.array(cost)

        return cost_function

    def teardown(self):
        print(f"log file saved at {self.log_path}")
        self.log_file.close()
