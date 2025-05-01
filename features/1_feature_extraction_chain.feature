Feature: Feature extraction pipelines
    Scenario: Process packets from a new offline pcap reader
        Given a new afterimage feature extraction pipeline initialized with test_data dataset and file benign_lenovo_bulb
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    Scenario Outline: Process packets from a loaded offline pcap reader
        Given a loaded afterimage feature extraction pipeline initialized with test_data dataset and file <file>
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

        Examples:
            | file                        |
            | malicious_ACK_Flooding      |
            | malicious_Service_Detection |
            | malicious_Port_Scanning     |
