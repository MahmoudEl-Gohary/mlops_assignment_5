import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

df = pd.read_csv('iris.csv')
X = df.drop('Id', axis=1, errors='ignore').drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("iris_validation_pipeline")

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    mlflow.log_metric("accuracy", accuracy)
    
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Training finished. Accuracy: {accuracy}")
    print(f"Run ID: {run_id} written to model_info.txt")