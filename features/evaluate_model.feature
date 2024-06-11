Feature: Evaluate Model
            I want to train a NIDS model

    Scenario Outline: Training NIDS
        Given A MultiLayerOCDModel with <model> as detection engine and <GNN> with <prior> as node encoder
        When we want to train with AfterImageGraph features from <filename> in <dataset> dataset with the following metrics:
        |metric_name |
        |detection_rate|
        Then The process should not report error

    Examples: Models
        | model | GNN            | prior    | filename           | dataset   |
        | AE    | GCNNodeEncoder | gaussian | sample_lenovo_bulb | Test_Data |