from behave import given, when, then

from ANIDSC.evaluator.summarizer import BasicSummarizer


@given('a Summarizer initialized with dataset')
def step_init_dataset(context):
    """
    Expects a table like:
      | dataset   |
      | test_data |
    """
    # Extract the single cell from the table
    context.dataset_names = [i['dataset'] for i in context.table.rows]

@given('initialized with feature extractor')
def step_init_feature_extractor(context):
    """
    Expects a table like:
      | feature_extractor |
      | AfterImage        |
    """
    context.fe_names= [i['feature_extractor'] for i in context.table.rows]
    
@when('we plot the results')
def step_when_plot_results(context):
    context.summarizer=BasicSummarizer(context.dataset_names, context.fe_names)
    context.summarizer.run()
    
@then("the plots are plotted")
def step_then_plots_plotted(context):
    # the pipeline should run
    assert context.failed is False