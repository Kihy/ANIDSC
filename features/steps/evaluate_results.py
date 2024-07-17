from behave import given, when, then

from ANIDSC.evaluations.summarizer import BasicSummarizer

@given('a Summarizer initialized with dataset "{dataset}" and files')
def step_given_summarizer(context, dataset):
    files = [row['file'] for row in context.table]
    summarizer=BasicSummarizer([dataset],files, calc_f1=True)
    context.summarizer=summarizer 

@when('we plot the results')
def step_when_plot_results(context):
    context.summarizer.plots()
    context.summarizer.gen_summary()
    
