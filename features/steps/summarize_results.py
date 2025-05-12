from behave import given, when, then

from ANIDSC.evaluator.summarizer import BasicSummarizer

@given('a Summarizer initialized with dataset {dataset} and feature extractor AfterImage')
def step_given_summarizer(context, dataset):
    
    summarizer=BasicSummarizer([dataset], ["AfterImageGraph","AfterImage"])
    context.summarizer=summarizer 

@when('we plot the results')
def step_when_plot_results(context):
    context.summarizer.run()
    
@then("the plots are plotted")
def step_then_plots_plotted(context):
    # the pipeline should run
    assert context.failed is False