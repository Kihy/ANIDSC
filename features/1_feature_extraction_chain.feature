Feature: Feature extraction pipelines

    Scenario: Empty folders
        Given test_data AfterImage folder is empty
        And test_data AfterImageGraph folder is empty

    Scenario Outline: Process packets from a offline pcap reader
        Given a <state> <fe_name> feature extraction pipeline initialized with test_data dataset and file <file>
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

        Examples:
            | state  | fe_name         | file                        |
            | new    | AfterImage      | benign_lenovo_bulb          |
            | loaded | AfterImage      | malicious_ACK_Flooding      |
            | loaded | AfterImage      | malicious_Service_Detection |
            | loaded | AfterImage      | malicious_Port_Scanning     |
            | new    | AfterImageGraph | benign_lenovo_bulb          |
            | loaded | AfterImageGraph | malicious_ACK_Flooding      |
            | loaded | AfterImageGraph | malicious_Service_Detection |
            | loaded | AfterImageGraph | malicious_Port_Scanning     |



