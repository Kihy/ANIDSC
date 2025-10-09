Feature: Detection with homogeneous models
    Scenario: Empty folders
        Given test_data NetworkAccessGraphExtractor folder is empty

    Scenario: Feature extraction
        Given The test_data file iterator
        And Meta Extractor: ProtocolMetaExtractor and Feature Extractor: NetworkAccessGraphExtractor
        And a graph_feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    Scenario Outline: Detection with homogeneous models, ICL and Kitsune does not work well with small dimension input
        Given The test_data file iterator
        And Meta Extractor: ProtocolMetaExtractor and Feature Extractor: AfterImage
        And Model: <model>
        And Node Embedder: <node_embed>
        And a homogeneous pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | model            | node_embed          |
            | torch_model.AE   | PassThroughEmbedder |
            | torch_model.GOAD | PassThroughEmbedder |
            | torch_model.SLAD | PassThroughEmbedder |
            | torch_model.VAE  | PassThroughEmbedder |
            | MedianDetector   | PassThroughEmbedder |
            | torch_model.AE   | GCNEmbedder         |
            | torch_model.GOAD | GCNEmbedder         |
            | torch_model.SLAD | GCNEmbedder         |
            | torch_model.VAE  | GCNEmbedder         |
            | MedianDetector   | GCNEmbedder         |
            | torch_model.AE   | GATEmbedder         |
            | torch_model.GOAD | GATEmbedder         |
            | torch_model.SLAD | GATEmbedder         |
            | torch_model.VAE  | GATEmbedder         |
            | MedianDetector   | GATEmbedder         |
            | torch_model.AE   | MLPEmbedder         |
            | torch_model.GOAD | MLPEmbedder         |
            | torch_model.SLAD | MLPEmbedder         |
            | torch_model.VAE  | MLPEmbedder         |
            | MedianDetector   | MLPEmbedder         |

