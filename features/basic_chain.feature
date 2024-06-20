Feature: Basic chaining of components

    Scenario: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and file "sample_lenovo_bulb"
        And a feature extraction pipeline with AfterImage
        And the output folder is empty
        And a basic NIDS model with AE
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged

    Scenario: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "sample_lenovo_bulb", and feature extractor "AfterImage"
        And a basic NIDS model with AE
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged