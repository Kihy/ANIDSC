Feature: Feature extraction pipelines
    Scenario Outline: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "test_data" and <file>
            And a feature extraction pipeline with AfterImage
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
    Examples:
        | file                        |
        | benign_lenovo_bulb          |
        | malicious_ACK_Flooding      |
        | malicious_Service_Detection |
        | malicious_Port_Scanning     |

    # Scenario Outline: Process packets from an offline pcap reader
    #     Given a PacketReader initialized with dataset "test_data" and <file>
    #         And a feature extraction pipeline with AfterImageGraph
    #     When the PacketReader starts
    #     Then the pipeline should not fail
    #         And the components are saved
    # Examples:
    #     | file                        |
    #     | benign_lenovo_bulb          |
    #     | malicious_Port_Scanning     |
    #     | malicious_Service_Detection |
    #     | malicious_ACK_Flooding      |