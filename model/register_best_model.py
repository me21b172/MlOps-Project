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
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parameters = dvc.api.params_show()

class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapper for the news classification model.
    Handles loading of PyTorch model and tokenizer, and provides prediction interface.
    """
    
    def load_context(self, context):
        """Load model artifacts and initialize tokenizer.
        
        Args:
            context: MLflow context object containing artifacts path
        """
        try:
            # Initialize device (GPU if available)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load the PyTorch model
            self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"]).to(self.device)
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_train.model_name)
            logger.info("Successfully loaded model and tokenizer")
            
        except Exception as e:
            logger.error(f"Error loading model context: {str(e)}")
            raise

    def predict(self, context, model_input):
        """Make predictions on input news text.
        
        Args:
            context: MLflow context object
            model_input: Either a list of strings or pandas DataFrame with 'text' column
            
        Returns:
            List of predicted labels
        """
        try:
            # Accept either list or DataFrame
            if isinstance(model_input, pd.DataFrame):
                news_list = model_input["text"].tolist()
            elif isinstance(model_input, list):
                news_list = model_input
            else:
                error_msg = "Input must be a list of strings or a pandas DataFrame with a 'text' column."
                logger.error(error_msg)
                raise ValueError(error_msg)

            return predict_helper(self.model, news_list)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
def run_mlflow_server():
    """Run MLflow server in a separate thread.
    The server will be automatically terminated when main program exits.
    """
    try:
        logger.info("Starting MLflow server...")
        cmd = ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
        process = subprocess.Popen(cmd)
        atexit.register(process.terminate)
        process.wait()  # Blocks until server terminates
    except Exception as e:
        logger.error(f"MLflow server failed: {str(e)}")
        raise

def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """Objective function for hyperparameter optimization.
    
    Args:
        params: Dictionary of hyperparameters to evaluate
        
    Returns:
        Dictionary containing optimization results including:
        - loss: Value to minimize (negative validation accuracy)
        - status: STATUS_OK if successful
        - val_accuracy: Validation accuracy
        - run_id: MLflow run ID for this trial
    """
    try:
        # Create descriptive run name
        run_name = f"neurons-{params['num_neurons']}_lr-{params['lr']:.0e}_batch-{params['batch_size']}_{datetime.datetime.now().strftime('%H%M%S')}"
        
        with mlflow.start_run(nested=True, run_name=run_name):
            logger.info(f"Starting trial with params: {params}")
            
            # Train model with current hyperparameters
            val_acc, model = build_model_train_model(
                num_neurons=params["num_neurons"], 
                lr=params["lr"], 
                batch_size=params["batch_size"], 
                num_epochs=params["num_epochs"]
            )
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("val_accuracy", val_acc)
            logger.info(f"Validation accuracy: {val_acc:.4f}")
            
            # Hyperopt minimizes the loss, so we use negative accuracy
            loss = -val_acc
            
            # Log the PyTorch model
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
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "loss": loss,
                "status": STATUS_OK,
                "val_accuracy": val_acc,
                "run_id": mlflow.active_run().info.run_id  # Store run ID for model registration
            }
            
    except Exception as e:
        logger.error(f"Trial failed with params {params}: {str(e)}")
        return {
            "loss": float('inf'),  # Return high loss for failed trials
            "status": STATUS_OK,  # Still return OK status to continue optimization
            "val_accuracy": 0.0,
            "run_id": None
        }

def main():
    """Main function to execute hyperparameter optimization.
    Sets up MLflow server, runs optimization, and registers best model.
    """
    try:
        # Start MLflow server in background thread
        server_thread = Thread(target=run_mlflow_server, daemon=True)
        server_thread.start()
        time.sleep(5)  # Give server time to start
        
        # Configure MLflow tracking
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("new-classifier-hyperopt")
        
        logger.info("Starting hyperparameter optimization...")
        
        with mlflow.start_run(run_name="hyperparameter_search"):
            trials = Trials()

            # Define search space from DVC parameters
            space = {
                "num_neurons": hp.choice("num_neurons", parameters["model"]["num_neurons"]),
                "lr": hp.choice("lr", parameters["model"]["lr"]),
                "batch_size": hp.choice("batch_size", parameters["model"]["batch_size"]),
                "num_epochs": hp.choice("num_epochs", parameters["model"]["epochs"])
            }

            # Run optimization
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
            
            # Add description to the registered model
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name="news_classifier",
                version=registered_model.version,
                description=f"Best model from hyperopt search. Val accuracy: {best_trial['val_accuracy']:.4f}"
            )
            
            logger.info(f"Best parameters: {best}")
            logger.info(f"Best validation accuracy: {best_trial['val_accuracy']}")
            logger.info(f"Registered model: {registered_model.name}, version {registered_model.version}")
            
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}")
        raise