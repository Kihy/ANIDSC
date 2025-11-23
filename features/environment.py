from datetime import datetime

def before_all(context):
    # This setup will run only once before any features or scenarios execute
    print("This setup runs only once before all features.")
    # Add your one-time step logic here
    context.run_identifier = "test_run"