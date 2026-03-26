import sys
import mlflow
import os

def check_accuracy():
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    
    if accuracy is None:
        print("Error: Accuracy metric not found in MLflow.")
        sys.exit(1)
        
    print(f"Evaluating Run ID: {run_id}")
    print(f"Recorded Accuracy: {accuracy}")
    
    if accuracy < 0.85:
        print("Validation Failed: Accuracy is strictly below the 0.85 threshold.")
        sys.exit(1)
    else:
        print("Validation Passed: Accuracy meets the requirement.")
        sys.exit(0)

if __name__ == "__main__":
    check_accuracy()