from ANIDSC.component.pipeline_component import Pipeline
from behave import given, when, then
from ANIDSC import model
from ANIDSC.component.evaluator import BaseEvaluator
from ANIDSC.feature_buffer.tabular import TabularFeatureBuffer


from ANIDSC.data_source.offline_sources import CSVReader, PacketReader
from ANIDSC.feature_extractor.after_image import AfterImage
from ANIDSC.feature_extractor.frequency import FrequencyExtractor

from shutil import rmtree
import os
import glob
from ANIDSC.component.model import BaseOnlineODModel
from ANIDSC.normalizer.t_digest import LivePercentile
from ANIDSC.templates import METRICS


@given('a PacketReader initialized with dataset "{dataset}" and {file}')
def step_given_packet_reader(context, dataset, file):
    # limit to 5e5 so it doesnt take too long
    context.data_source = PacketReader(dataset, file, max_records=1e5)
    context.dataset = dataset
    context.file_name = file




@given("a feature extraction pipeline with frequency analysis")
def step_given_feature_extraction_with_frequency_analysis(context):
    feature_extractor = FrequencyExtractor()
    context.fe_name = "FrequencyExtractor"
    feature_buffer = TabularFeatureBuffer(buffer_size=256)
    context.pipeline = feature_extractor | feature_buffer


@given("a basic boxplot model")
def step_given_basic_boxplot_model(context):
    model = getattr(model, "BoxPlot")()

    evaluator = BaseEvaluator(
        METRICS,
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    if hasattr(context, "pipeline"):
        context.pipeline = context.pipeline | model | evaluator
    else:
        context.pipeline = model | evaluator
    context.model_name = "BoxPlot"


@given("the output folder is empty")
def step_given_output_folder_is_empty(context):
    file_dir = f"{context.dataset}/{context.fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)

    file_dir = f"{context.dataset}/{context.fe_name}/runs"
    if os.path.exists(file_dir):
        rmtree(file_dir)


@given(
    'a csv file initialized with dataset "{dataset}", file "{file_name}", and feature extractor AfterImage'
)
def step_given_csv_file(context, dataset, file_name):
    fe_name = "AfterImage"
    context.data_source = CSVReader(fe_name, fe_name, dataset_name=dataset, file_name=file_name)
    context.dataset = dataset
    context.file_name = file_name
    context.fe_name = fe_name
    context.pipeline_components = ["detection"]
    context.benign_file="benign_lenovo_bulb"

@given("a {state} basic pipeline with {model_name}")
def step_given_basic_NIDS_model_with_AE(context,state, model_name):
    if state == "new":
        scaler=LivePercentile()
        detector=BaseOnlineODModel(f"ANIDSC.models.{model_name}")
        evaluator=BaseEvaluator(METRICS)
        
        context.pipeline = scaler|detector|evaluator
    else:
        scaler=LivePercentile.load(context.dataset, context.fe_name, context.benign_file, "LivePercentile")
        detector=BaseOnlineODModel.load(context.dataset, context.fe_name, context.benign_file, "BaseOnlineODModel")
        evaluator=BaseEvaluator(METRICS)
        
        context.pipeline = scaler|detector|evaluator
    
    

    

    context.model_name = model_name


@when("the PacketReader starts")
def step_when_packet_reader_starts(context):
    context.data_source >> context.pipeline
    context.pipeline.start()



@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    manifest_path=context.pipeline.get_save_path("yaml")
    
    loaded_pipeline=Pipeline.load(manifest_path)
    
    assert loaded_pipeline==context.pipeline
    
    


@then("the results are logged")
def step_then_results_logged(context):
    feature_path = f"{context.dataset}/{context.fe_name}/features/{context.file_name}.csv"
    assert os.path.isfile(feature_path)

    meta_path = f"{context.dataset}/{context.fe_name}/metadata/{context.file_name}.csv"
    assert os.path.isfile(meta_path)

    result_path = f"{context.dataset}/{context.fe_name}/results/{context.file_name}/{context.pipeline}.csv"
    assert os.path.isfile(result_path)

    tensorboard_path = (
        f"{context.dataset}/{context.fe_name}/runs/{context.file_name}/{context.pipeline}"
    )
    assert os.path.isdir(tensorboard_path)
