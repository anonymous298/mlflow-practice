from mlflow import MlflowClient

client = MlflowClient(tracking_uri='http://127.0.0.1:8080')

# Creating Experiment
experiment_id = client.create_experiment('My Experiment')
print(experiment_id)

# log data to an experiment
run = client.create_run(experiment_id)
client.log_param(run.info.run_id, "learning rate", 0.0001)
client.log_metric(run.info.run_id, "accuracy", 0.98)