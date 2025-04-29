Feature: End to End chaining from PCAP readers
    Scenario: Process packets directly from an offline pcap reader with basic pipeline
        Given a PacketReader initialized with dataset "test_data" and benign_lenovo_bulb
            And a feature extraction pipeline with AfterImage
            And a new basic pipeline with AE
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged

    Scenario: Process packets with boxplot
        Given a PacketReader initialized with dataset "test_data" and benign_lenovo_bulb
            And a feature extraction pipeline with frequency analysis
            And a basic boxplot model
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged

    Scenario Outline: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "test_data" and <file>
            And a feature extraction pipeline with AfterImage
            And a new CDD pipeline with ARCUS over AE
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | file                        |
        | benign_lenovo_bulb          |
        | malicious_ACK_Flooding      |
        | malicious_Service_Detection |
        | malicious_Port_Scanning     |