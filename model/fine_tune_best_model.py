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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_serving.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mappings for classification
label_map = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
reverse_map = {0: 'business', 1: 'entertainment', 2: 'politics', 3: 'sport', 4: 'tech'}

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model configuration
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()

def get_latest_model_version():
    """
    Retrieve the latest version of the 'news_classifier' model from MLflow.
    
    Returns:
        ModelVersion: The latest model version object
        
    Raises:
        Exception: If no model versions are found
    """
    try:
        versions = client.search_model_versions(
            "name='news_classifier'", 
            order_by=["version_number DESC"], 
            max_results=1
        )
        if not versions:
            raise Exception("No model versions found for 'news_classifier'")
        return versions[0]
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        raise

def expose_best_model(run):
    """
    Start MLflow model serving as a background process.
    
    Args:
        run: Model version object to serve
        
    Returns:
        subprocess.Popen: Process object for the serving instance
        
    Raises:
        subprocess.SubprocessError: If process fails to start
    """
    try:
        model_uri = f"models:/{run.name}/{run.version}"
        cmd = ["mlflow", "models", "serve", "-m", model_uri, "--host", "0.0.0.0", "--port", "5002", "--no-conda"]
        logger.info(f"Starting model serving with command: {' '.join(cmd)}")
        return subprocess.Popen(cmd)
    except Exception as e:
        logger.error(f"Failed to start model serving: {e}")
        raise

def is_port_in_use(port: int) -> bool:
    """
    Check if a network port is currently in use.
    
    Args:
        port: Port number to check
        
    Returns:
        bool: True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """
    Kill any process running on the specified port (Windows specific).
    
    Args:
        port: Port number to clear
        
    Raises:
        Exception: If process killing fails
    """
    try:
        logger.info(f"Attempting to kill process on port {port}")
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
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid != "0":  # Skip critical system PID 0
                        logger.info(f"Killing process {pid} on port {port}")
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                    else:
                        logger.warning(f"Skipping PID 0 (system process) on port {port}")
        else:
            logger.info(f"No process found on port {port}")
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")
        raise

def extract_data():
    """
    Extract and preprocess news data from CSV file.
    
    Returns:
        pandas.DataFrame: Processed data with text and mapped labels
        
    Raises:
        FileNotFoundError: If data file is missing
        ValueError: If data format is invalid
    """
    try:
        logger.info("Extracting data from news_feed.csv")
        data = pd.read_csv("data/news_feed.csv")
        data = pd.concat([data["Text"], data["Category"]], axis=1)
        data['Category'] = data['Category'].map(model_train.label_map)
        kill_process_on_port(5002)
        logger.info(f"Extracted {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise

def extract_params(latest_version):
    """
    Extract parameters from an MLflow run.
    
    Args:
        latest_version: Model version object
        
    Returns:
        dict: Dictionary of run parameters
        
    Raises:
        mlflow.exceptions.MlflowException: If run cannot be accessed
    """
    try:
        run = client.get_run(latest_version.run_id)
        logger.debug(f"Extracted params from run {latest_version.run_id}")
        return run.data.params
    except Exception as e:
        logger.error(f"Error extracting parameters: {e}")
        raise

def extract_unwrapped_model(latest_version):
    """
    Load the underlying PyTorch model from MLflow.
    
    Args:
        latest_version: Model version object
        
    Returns:
        torch.nn.Module: The unwrapped PyTorch model
        
    Raises:
        mlflow.exceptions.MlflowException: If model loading fails
    """
    try:
        model_uri = f"models:/{latest_version.name}/1"
        logger.info(f"Loading model from URI: {model_uri}")
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        return pyfunc_model._model_impl.python_model.model
    except Exception as e:
        logger.error(f"Error loading unwrapped model: {e}")
        raise

def predict_helper(model, news: list):
    """
    Helper function for making predictions with the model.
    
    Args:
        model: Loaded PyTorch model
        news: List of news texts to classify
        
    Returns:
        list: List of predicted categories
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        inputs = tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        label = torch.argmax(outputs.logits, dim=1)
        return [reverse_map[label.data[i].item()] for i in range(len(label))]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for the news classifier.
    Handles model loading and prediction interface.
    """
    
    def load_context(self, context):
        """
        Load model artifacts when the wrapper is initialized.
        
        Args:
            context: MLflow context object containing artifacts
        """
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"]).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Successfully loaded model in wrapper")
        except Exception as e:
            logger.error(f"Error loading model context: {e}")
            raise

    def predict(self, context, model_input):
        """
        Make predictions on input data.
        
        Args:
            context: MLflow context
            model_input: Input data (list or DataFrame)
            
        Returns:
            list: Predicted categories
            
        Raises:
            ValueError: For invalid input types
        """
        try:
            if isinstance(model_input, pd.DataFrame):
                news_list = model_input["text"].tolist()
            elif isinstance(model_input, list):
                news_list = model_input
            else:
                raise ValueError("Input must be a list of strings or a pandas DataFrame with a 'text' column.")
            
            return predict_helper(self.model, news_list)
        except Exception as e:
            logger.error(f"Prediction failed in wrapper: {e}")
            raise

def run_mlflow_server():
    """
    Run MLflow tracking server as a background process.
    """
    try:
        cmd = ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
        logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        atexit.register(process.terminate)
        process.wait()
    except Exception as e:
        logger.error(f"MLflow server failed: {e}")
        raise

def main():
    """
    Main execution function for model training and serving workflow.
    """
    try:
        logger.info("Starting main execution")
        
        # Start MLflow server if not running
        if not is_port_in_use(8080):
            server_thread = Thread(target=run_mlflow_server, daemon=True)
            server_thread.start()
            time.sleep(5)  # Give server time to start
        
        mlflow.set_experiment("new-classifier-hyperopt")
        latest_version = get_latest_model_version()
        logger.info(f"Using model version: {latest_version.version}")
        
        with mlflow.start_run(run_name="hyperparameter_search"):
            # Load data, params and model
            data = extract_data()
            params = extract_params(latest_version)
            model = extract_unwrapped_model(latest_version)
            
            # Convert params to proper types
            params['num_neurons'] = int(params['num_neurons'])
            params['lr'] = float(params['lr'])
            params['batch_size'] = int(params['batch_size'])
            
            run_name = f"neurons-{params['num_neurons']}_lr-{params['lr']:.0e}_batch-{params['batch_size']}_finetuning_{datetime.datetime.now().strftime('%H%M%S')}"
            
            with mlflow.start_run(nested=True, run_name=run_name) as run:
                # Train and evaluate model
                predictions = train_model(
                    model=model,
                    data=data,
                    lr=params["lr"],
                    batch_size=params["batch_size"],
                    num_epoch=1
                )
                
                # Log results
                mlflow.log_params(params=params)
                mlflow.log_metric("val_accuracy", predictions)
                
                # Create model signature
                signature = infer_signature(
                    model_input=["Sample news text", "Sachin Tendulkar is the best player in the world"],
                    model_output=["business", "sport"]
                )
                
                # Log and register model
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
                
                # Update model version description
                client.update_model_version(
                    name="news_classifier",
                    version=registered_model.version,
                    description=f"Best model from hyperopt search. Val accuracy: {predictions:.4f}"
                )
                
                logger.info(f"Registered new model version {registered_model.version} with accuracy {predictions:.4f}")
                
    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
        latest_version = get_latest_model_version()
        logger.info(f"Latest model version: {latest_version.version}")
        logger.info(f"Run ID: {latest_version.run_id}")
        
        mlflow_server = expose_best_model(latest_version)
        logger.info(f"MLflow model server started with PID: {mlflow_server.pid}")
        
        # Clear the data file
        df = pd.read_csv("data/news_feed.csv")
        df.head(0).to_csv("data/news_feed.csv", index=False)
        logger.info("Cleared news_feed.csv")
        
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)