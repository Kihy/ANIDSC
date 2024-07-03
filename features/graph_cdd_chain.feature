Feature: NIDS Chain with Graph Representation and conceprt drift detection

    Scenario: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and file "sample_lenovo_bulb"
        And a feature extraction pipeline with AfterImageGraph
        And the output folder is empty
        And a graph NIDS concept drift detection model with AE
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged

    Scenario: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "sample_lenovo_bulb", and feature extractor "AfterImageGraph"
        And a graph NIDS concept drift detection model with AE
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged