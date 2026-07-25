"""This piece of code helps to train the model using XGBoost
and log the metrics on MLflow"""

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score
from xgboost import XGBClassifier

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

# Start a tracked MLflow run named "xgboost"
with mlflow.start_run(run_name="xgboost"):

    # Create and train the model
    # scale_pos_weight compensates for class imbalance
    # (ratio of healthy:diseased ≈ 4265:74)
    model = XGBClassifier(scale_pos_weight=(4265/74), eval_metric="logloss",
                          random_state=42)
    model.fit(X_train, y_train)

    # Predict on unseen test data and score the predictions
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # Log settings and results to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("recall", recall)

    # Save the trained model itself (XGBoost-specific format)
    # for later registration/serving
    mlflow.xgboost.log_model(model, "model")

    # Print a full readable performance breakdown to the terminal
    print(classification_report(y_test, preds))
