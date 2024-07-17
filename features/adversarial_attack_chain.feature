Feature: An attack chain that produces adversarial packets in real time
    Scenario: Conduct adversarial attack on packets originating from offline pcap file
            Given a PacketReader initialized with dataset "../datasets/Test_Data" and malicious_Port_Scanning
                And a Liuer Mihou attack with trained 'AE' model and AfterImage feature extractor from 'benign_lenovo_bulb'
            When the PacketReader starts
            Then the pipeline should not fail
                