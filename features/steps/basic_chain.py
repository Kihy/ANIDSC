from behave import given, when, then
from ANIDSC.base_files.evaluator import BaseEvaluator
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.base_files.model import ConceptDriftWrapper
from ANIDSC.data_source.offline_sources import CSVReader, PacketReader
from ANIDSC.feature_extractors.after_image import AfterImage
from ANIDSC.models.autoencoder import AE
from ANIDSC.normalizer.t_digest import LivePercentile
from shutil import rmtree
import os 
import glob

@given('a PacketReader initialized with dataset "{dataset}" and file "{file}"')
def step_given_packet_reader(context, dataset, file):
    context.data_source = PacketReader(dataset, file)
    context.dataset=dataset
    context.file=file
    
@given('a feature extraction pipeline with AfterImage')
def step_given_feature_extraction_pipeline_with_AfterImage(context):
    feature_extractor = AfterImage()
    context.fe_name="AfterImage"
    feature_buffer = FeatureBuffer(buffer_size=256)
    context.feature_extraction = feature_extractor | feature_buffer
    
@given('the output folder is empty')
def step_given_output_folder_is_empty(context):
    file_dir=f"{context.dataset}/{context.fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)
        
    file_dir=f"{context.dataset}/runs"
    if os.path.exists(file_dir):
        rmtree(file_dir)
        
@given('a csv file initialized with dataset "{dataset}", file "{file}", and feature extractor "{fe_name}"')
def step_given_csv_file(context, dataset, file, fe_name):
    if fe_name=="AfterImageGraph":
        context.protocols={"TCP":0,"UDP":1}
    else:
        context.protocols={}
    context.data_source = CSVReader(dataset, fe_name, file, fe_features=100, protocols=context.protocols)
    context.dataset=dataset
    context.file=file
    context.fe_name=fe_name

    


@given('a basic NIDS model with AE')
def step_given_basic_NIDS_model_with_AE(context):
    model = AE(
        preprocessors=["to_float_tensor", "to_device"],
    )
    standardizer = LivePercentile()
   
    evaluator = BaseEvaluator(
        ["detection_rate", "average_score", "average_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=10,
    )
    context.detection = standardizer | model | evaluator
    context.model_name="AE"
    

@when('the PacketReader starts')
def step_when_packet_reader_starts(context):
    
    if hasattr(context, "feature_extraction"):
        context.pipeline = context.feature_extraction | context.detection
    else:
        context.pipeline=context.detection
    
    context.data_source >> context.pipeline
    context.data_source.start()

@then('the pipeline should not fail')
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False
    
@then('the components are saved')
def step_then_components_are_saved(context):
    for component in context.pipeline.components:
        print(component)
        if component.component_type != "":
            
            path=f"{context.dataset}/{context.fe_name}/{component.component_type}/{context.file}/{component.name}.*"
            assert glob.glob(path)

    
@then('the results are logged')
def step_then_results_logged(context):
    feature_path=f"{context.dataset}/{context.fe_name}/features/{context.file}.csv"
    assert os.path.isfile(feature_path)
    
    meta_path=f"{context.dataset}/{context.fe_name}/metadata/{context.file}.csv"
    assert os.path.isfile(meta_path)
    
    result_path=f"{context.dataset}/{context.fe_name}/results/{context.file}/{context.model_name}.csv"
    assert os.path.isfile(result_path)
    
    tensorboard_path=f"{context.dataset}/runs/{context.pipeline}"
    assert os.path.isdir(tensorboard_path)
    