Feature: Detection with homogeneous models
    Scenario: Meta extraction
        Given test_data ProtocolMetaExtractor folder is empty
        Given The test_data file iterator
        And Data Source: PacketReader
        And Meta Extractor: ProtocolMetaExtractor
        And a meta_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario: Feature extraction
        Given test_data NetworkAccessGraphExtractor folder is empty
        Given The test_data file iterator
        And Data Source: CSVReader
        And FE Name: ProtocolMetaExtractor
        And Feature Extractor: NetworkAccessGraphExtractor
        And a feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    Scenario Outline: Detection with homogeneous models, ICL and Kitsune does not work well with small dimension input
        Given The test_data file iterator
        And FE Name: NetworkAccessGraphExtractor
        And Data Source: JsonGraphReader
        And Graph Rep: <graph_rep>
        And Model: <model>
        And Node Embedder: <node_embed>
        And a homogeneous pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | model            | node_embed          | graph_rep |
            | torch_model.AE   | MLPEmbedder         | CDD       |
            | MedianDetector   | MLPEmbedder         | CDD       |
            | torch_model.GOAD | MLPEmbedder         | CDD       |
            | torch_model.SLAD | MLPEmbedder         | CDD       |
            | torch_model.VAE  | MLPEmbedder         | CDD       |
            | torch_model.AE   | PassThroughEmbedder | CDD       |
            | torch_model.GOAD | PassThroughEmbedder | CDD       |
            | torch_model.SLAD | PassThroughEmbedder | CDD       |
            | torch_model.VAE  | PassThroughEmbedder | CDD       |
            | MedianDetector   | PassThroughEmbedder | CDD       |
            | torch_model.AE   | GATEmbedder         | CDD       |
            | torch_model.GOAD | GATEmbedder         | CDD       |
            | torch_model.SLAD | GATEmbedder         | CDD       |
            | torch_model.VAE  | GATEmbedder         | CDD       |
            | MedianDetector   | GATEmbedder         | CDD       |
            | torch_model.AE   | PassThroughEmbedder | Plain     |
            | torch_model.GOAD | PassThroughEmbedder | Plain     |
            | torch_model.SLAD | PassThroughEmbedder | Plain     |
            | torch_model.VAE  | PassThroughEmbedder | Plain     |
            | MedianDetector   | PassThroughEmbedder | Plain     |
            | torch_model.AE   | MLPEmbedder         | Plain     |
            | torch_model.GOAD | MLPEmbedder         | Plain     |
            | torch_model.SLAD | MLPEmbedder         | Plain     |
            | torch_model.VAE  | MLPEmbedder         | Plain     |
            | MedianDetector   | MLPEmbedder         | Plain     |
            | torch_model.AE   | GCNEmbedder         | Filter    |
            | torch_model.GOAD | GCNEmbedder         | Filter    |
            | torch_model.SLAD | GCNEmbedder         | Filter    |
            | torch_model.VAE  | GCNEmbedder         | Filter    |
            | MedianDetector   | GCNEmbedder         | Filter    |
            | torch_model.AE   | MLPEmbedder         | Filter    |
            | torch_model.GOAD | MLPEmbedder         | Filter    |
            | torch_model.SLAD | MLPEmbedder         | Filter    |
            | torch_model.VAE  | MLPEmbedder         | Filter    |
            | MedianDetector   | MLPEmbedder         | Filter    |

