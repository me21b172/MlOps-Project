import gc
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import datetime
import torch
from transformers import AutoTokenizer
from mlflow.tracking import MlflowClient
import mlflow
from mlflow.models import infer_signature
from model_train import train_model, build_model_train_model, predict_helper
import subprocess
import time
import atexit
from threading import Thread
import model_train
import dvc.api

parameters = dvc.api.params_show()

class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # load the raw torch model artifact
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"]).to(self.device)
        # re‑instantiate your tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_train.model_name)

    def predict(self, context, model_input):
        # Accept either list or DataFrame
        if isinstance(model_input, pd.DataFrame):
            news_list = model_input["text"].tolist()
        elif isinstance(model_input, list):
            news_list = model_input
        else:
            raise ValueError("Input must be a list of strings or a pandas DataFrame with a 'text' column.")

        return predict_helper(self.model, news_list)
    
def run_mlflow_server():
    """Thread target function for server"""
    cmd = ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
    process = subprocess.Popen(cmd)
    atexit.register(process.terminate)
    process.wait()  # Blocks until server terminates

# Start in dedicated thread

def objective(params):
    run_name = f"neurons-{params['num_neurons']}_lr-{params['lr']:.0e}_batch-{params['batch_size']}_{datetime.datetime.now().strftime('%H%M%S')}"
    with mlflow.start_run(nested=True, run_name=run_name):
        val_acc, model = build_model_train_model(
            num_neurons=params["num_neurons"], 
            lr=params["lr"], 
            batch_size=params["batch_size"], 
            num_epochs=params["num_epochs"]
        )
        mlflow.log_params(params)
        mlflow.log_metric("val_accuracy", val_acc)
        loss = -val_acc
        
        # Log the model
        signature = infer_signature(
            model_input=["Sample news text","Sachin Tendulkar is the best player in the world"],
            model_output=["business","sport"]
        )
        mlflow.pytorch.log_model(model, "pytorch_model")
        # Log pyfunc wrapper pointing to PyTorch model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=NewsClassifierWrapper(),
            artifacts={"pytorch_model": f"runs:/{mlflow.active_run().info.run_id}/pytorch_model"},
)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "loss": loss,
            "status": STATUS_OK,
            "val_accuracy": val_acc,
            "run_id": mlflow.active_run().info.run_id  # Store run ID instead of model
        }

def main():
    server_thread = Thread(target=run_mlflow_server, daemon=True)
    server_thread.start()
    time.sleep(5)  # Give server time to start
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("new-classifier-hyperopt")
    with mlflow.start_run(run_name="hyperparameter_search"):
        trials = Trials()

        space = {
            "num_neurons": hp.choice("num_neurons",parameters["model"]["num_neurons"]),
            "lr": hp.choice("lr", parameters["model"]["lr"]),
            "batch_size": hp.choice("batch_size",parameters["model"]["batch_size"]),
            "num_epochs": hp.choice("num_epochs", parameters["model"]["epochs"])
        }

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=parameters["model"]["runs"],
            trials=trials,
        )

        # Get the best trial (lowest loss)
        best_trial = min(trials.results, key=lambda x: x["loss"])
        
        # Log the best parameters and metrics
        mlflow.log_params(best)
        mlflow.log_metric("best_val_accuracy", best_trial["val_accuracy"])
        
        # Register the best model
        model_uri = f"runs:/{best_trial['run_id']}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="news_classifier",
        )
        
        # Optional: Add description and transition to production
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name="news_classifier",
            version=registered_model.version,
            description=f"Best model from hyperopt search. Val accuracy: {best_trial['val_accuracy']:.4f}"
        )
        
        print(f"Best parameters: {best}")
        print(f"Best validation accuracy: {best_trial['val_accuracy']}")
        print(f"Registered model: {registered_model.name}, version {registered_model.version}")

if __name__ == "__main__":
    main()