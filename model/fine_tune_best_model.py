from mlflow.tracking import MlflowClient
import mlflow
from model_train import train_model
import pandas as pd
from mlflow.models import infer_signature
import datetime
import model_train
import subprocess
import socket
import time
from threading import Thread
import atexit
import torch
from transformers import AutoTokenizer
import os
import signal

label_map = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
reverse_map = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}
# Load model in half precision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()
def get_latest_model_version():
    return client.search_model_versions(
        "name='news_classifier'", 
        order_by=["version_number DESC"], 
        max_results=1
    )[0]

def expose_best_model(run):
    """Start MLflow server as background process"""
    model_uri = f"models:/{run.name}/{run.version}"
    # mlflow models serve -m "models:/wine-quality/1" --port 5002
    cmd = ["mlflow", "models" ,"serve", "-m", model_uri,"--host", "0.0.0.0", "--port", "5002", "--no-conda"]
    return subprocess.Popen(cmd)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    try:
        # Use netstat and taskkill for Windows
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        if output:
            lines = output.strip().split("\n")
            for line in lines:
                parts = line.split()
                pid = parts[-1]
                print(f"Killing process {pid} on port {port}")
                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
        else:
            print(f"No process found on port {port}")
    except Exception as e:
        print(f"Error: {e}")


def extract_data():
    data = pd.read_csv(f"news_feed.csv")
    data = pd.concat([data["Text"],data["Category"]],axis=1)
    data['Category'] = data['Category'].map(model_train.label_map)
    kill_process_on_port(5002)
    # df = pd.read_csv("news_feed.csv")
    # df.head(0).to_csv("news_feed.csv", index=False)
    return data

def extract_params(latest_version):
    run = client.get_run(latest_version.run_id)
    return run.data.params

# def extract_unwrapped_model(latest_version):
#     model_uri = f"models:/{latest_version.name}/{latest_version.version}"
#     # Load as pyfunc first
#     pyfunc_model = mlflow.pyfunc.load_model(model_uri)
#     # Access the underlying PyTorch model
#     return pyfunc_model._model_impl.model  
def extract_unwrapped_model(latest_version):
    # model_uri = f"models:/{latest_version.name}/{latest_version.version}"
    model_uri = f"models:/{latest_version.name}/1"
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    # Correct access chain through MLflow's internal wrapper
    return pyfunc_model._model_impl.python_model.model

def main():
    if not is_port_in_use(8080):
        server_thread = Thread(target=run_mlflow_server, daemon=True)
        server_thread.start()
        time.sleep(5)  # Give server time to start
    mlflow.set_experiment("new-classifier-hyperopt")
    print(get_latest_model_version())
    with mlflow.start_run(run_name="hyperparameter_search"):
        latest_version = get_latest_model_version()
        data,params,model = extract_data(),extract_params(latest_version),extract_unwrapped_model(latest_version)
        params['num_neurons'],params['lr'],params['batch_size'] = int(params['num_neurons']),float(params['lr']),int(params['batch_size'])
        run_name = f"neurons-{params['num_neurons']}_lr-{params['lr']:.0e}_batch-{params['batch_size']}_finetuning_{datetime.datetime.now().strftime('%H%M%S')}"
        with mlflow.start_run(nested=True, run_name=run_name) as run:
            predictions = train_model(model=model,data=data,lr=params["lr"],batch_size=params["batch_size"],num_epoch=1)
            mlflow.log_params(params=params)
            mlflow.log_metric("val_accuracy",predictions)
            # Log the model
            signature = infer_signature(
                model_input=["Sample news text","Sachin Tendulkar is the best player in the world"],
                model_output=["business","sport"]
            )
            mlflow.pytorch.log_model(model, "pytorch_model")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=NewsClassifierWrapper(),
                artifacts={"pytorch_model": f"runs:/{run.info.run_id}/pytorch_model"},
                signature=signature
            )
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="news_classifier"
            )
            
            # Optional: Add description and transition to production
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name="news_classifier",
                version=registered_model.version,
                description=f"Best model from hyperopt search. Val accuracy: {predictions:.4f}"
            )

def predict_helper(model,news:list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    model.eval()
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    label = torch.argmax(outputs.logits, dim=1)
    return [reverse_map[label.data[i].item()] for i in range(len(label))]

class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # load the raw torch model artifact
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"]).to(self.device)
        # re‑instantiate your tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

if __name__ == "__main__":
    kill_process_on_port(5002)
    main()
    print(get_latest_model_version().version)
    print(get_latest_model_version().run_id)
    mlflow_server = expose_best_model(get_latest_model_version())
    print(f"MLflow server started with PID: {mlflow_server.pid}")
    df = pd.read_csv("news_feed.csv")
    df.head(0).to_csv("news_feed.csv", index=False)