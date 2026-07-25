"""This piece of code helps to train the model
using Logistic Regression
and log the metrics on MLflow"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score

# Store MLflow data as plain files in ./mlruns
# (git-friendly, works consistently
# both locally and in GitHub Actions, instead of the local SQLite database)
mlflow.set_tracking_uri("file:./mlruns")

# Load training and testing data
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/testing.csv")

# Split into features (X) and target (y); convert labels to numbers
# (n=healthy=0, w=diseased=1)
X_train = train.drop(columns=["class"])
y_train = train["class"].map({"n": 0, "w": 1})
X_test = test.drop(columns=["class"])
y_test = test["class"].map({"n": 0, "w": 1})

# Group all runs under this experiment name in MLflow
mlflow.set_experiment("wilt-classification")

# Start a tracked MLflow run named "baseline_logreg"
with mlflow.start_run(run_name="baseline_logreg"):

    # Create and train the model; class_weight="balanced"
    # compensates for the rare diseased class
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on unseen test data and score the predictions
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # Log the settings used for this run
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")

    # Log the performance results for this run
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("recall", recall)

    # Save the trained model itself so it can be registered/served later
    mlflow.sklearn.log_model(model, "model")

    # Print a full readable performance breakdown to the terminal
    print(classification_report(y_test, preds))
