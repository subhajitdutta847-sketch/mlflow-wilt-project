"""Retrains an improved model, compares it to the current production model,
and promotes it automatically if it performs better.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Client used to fetch/compare/promote models in the registry
client = mlflow.tracking.MlflowClient()

# Load training and testing data
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/testing.csv")

# Split into features (X) and target (y); convert labels to numbers
# (n=healthy=0, w=diseased=1)
X_train = train.drop(columns=["class"])
y_train = train["class"].map({"n": 0, "w": 1})
X_test = test.drop(columns=["class"])
y_test = test["class"].map({"n": 0, "w": 1})

# Group this run under the same experiment as the earlier training runs
mlflow.set_experiment("wilt-classification")

# Train an improved model: same algorithm, but with feature scaling added
with mlflow.start_run(run_name="logreg_v2_scaled"):
    # Pipeline: scale features first, then feed them into logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(X_train, y_train)

    # Predict on test data and measure recall (our priority metric)
    preds = model.predict(X_test)
    new_recall = recall_score(y_test, preds)

    # Log settings, results, and the trained model to MLflow
    mlflow.log_param("model_type", "LogisticRegression_scaled")
    mlflow.log_metric("recall", new_recall)
    mlflow.sklearn.log_model(model, "model")

    # Save this run's ID so we can reference its model later if it wins
    new_run_id = mlflow.active_run().info.run_id

print(f"New model recall: {new_recall:.4f}")

# Fetch the current production model's recall for comparison
prod_version = client.get_model_version_by_alias("wilt-model", "production")
prod_run = client.get_run(prod_version.run_id)
prod_recall = prod_run.data.metrics["recall"]

print(f"Current production recall: {prod_recall:.4f}")

# Promote the new model only if it beats the current production model
if new_recall > prod_recall:
    MODEL_URI = f"runs:/{new_run_id}/model"
    # Register the new model as the next version of "wilt-model"
    result = mlflow.register_model(MODEL_URI, "wilt-model")
    # Reassign the "production" alias to this new version —
    # no manual step needed
    client.set_registered_model_alias("wilt-model", "production",
                                      result.version)
    print(f"Promoted new model as version {result.version} (production).")
else:
    # New model wasn't better, so production stays as-is
    print("New model did not beat production. No promotion.")
