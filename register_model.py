"""This piece of code helps to register the model on MLflow"""

import mlflow

# Client used to search runs and manage the model registry
client = mlflow.tracking.MlflowClient()

# Find the most recent run named "baseline_logreg" (our chosen best model)
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.mlflow.runName = 'baseline_logreg'",
    order_by=["start_time DESC"],
    max_results=1
)

# Get that run's ID and build the path to its saved model artifact
run_id = runs[0].info.run_id
MODEL_URI = f"runs:/{run_id}/model"

# Register that model under the name "wilt-model" in the Model Registry
result = mlflow.register_model(MODEL_URI, "wilt-model")
print(f"Registered version: {result.version}")

# Tag this version with alias "production"
# marks it as the officially active model
client.set_registered_model_alias("wilt-model", "production", result.version)
print("Alias 'production' set.")
