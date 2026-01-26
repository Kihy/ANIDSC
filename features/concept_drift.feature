Feature: Detection with homogeneous models
    Scenario: Empty folders
    Given folders that starts with test_data SingleLayerGraphExtractor are empty
    Given folders that starts with test_data ProtocolMetaExtractor are empty

    Scenario: Meta extraction
        Given The test_data file iterator
        And Data Source: PacketReader
        And Meta Extractor: ProtocolMetaExtractor
        And a meta_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario Outline: Feature extraction
        Given The test_data file iterator
        And Data Source: CSVReader
        And FE Name: ProtocolMetaExtractor
        And Feature Extractor: SingleLayerGraphExtractor
        And FE attributes: <layer>
        And a feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | layer     |
            | transport |
            | internet  |
            | physical  |


    Scenario Outline: Detection with homogeneous models, ICL and Kitsune does not work well with small dimension input
        Given The test_data file iterator
        And FE Name: SingleLayerGraphExtractor(<layer>)
        And Data Source: JsonGraphReader
        And Graph Rep: <graph_rep>
        And Model: <model>
        And Node Embedder: <node_embed>
        And a homogeneous pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model          | node_embed          | graph_rep | layer     |
            | torch_model.AE | PassThroughEmbedder | Plain     | physical  |
            | MedianDetector | PassThroughEmbedder | Plain     | physical  |
            | torch_model.AE | PassThroughEmbedder | CDD       | physical  |
            | MedianDetector | PassThroughEmbedder | CDD       | physical  |
            | torch_model.AE | PassThroughEmbedder | Filter    | physical  |
            | MedianDetector | PassThroughEmbedder | Filter    | physical  |
            | torch_model.AE | PassThroughEmbedder | Plain     | transport |
            | MedianDetector | PassThroughEmbedder | Plain     | transport |
            | torch_model.AE | PassThroughEmbedder | CDD       | transport |
            | MedianDetector | PassThroughEmbedder | CDD       | transport |
            | torch_model.AE | PassThroughEmbedder | Filter    | transport |
            | MedianDetector | PassThroughEmbedder | Filter    | transport |
            | torch_model.AE | PassThroughEmbedder | Plain     | internet  |
            | MedianDetector | PassThroughEmbedder | Plain     | internet  |
            | torch_model.AE | PassThroughEmbedder | CDD       | internet  |
            | MedianDetector | PassThroughEmbedder | CDD       | internet  |
            | torch_model.AE | PassThroughEmbedder | Filter    | internet  |
            | MedianDetector | PassThroughEmbedder | Filter    | internet  |

