import subprocess
import atexit
import time
from threading import Thread
import pandas as pd
from ruamel.yaml import YAML
from datetime import date
import logging

# Global variables
yaml = YAML()
file_path = "params.yaml"
pipeline_process = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
yaml = YAML()
file_path = "params.yaml"
pipeline_process = None

def run_mlflow_server():
    """
    Start MLflow server in a new terminal window.
    
    Raises:
        subprocess.SubprocessError: If failed to start MLflow server.
    """
    try:
        cmd = ["start", "cmd", "/k", "mlflow server --host 0.0.0.0 --port 8080"]
        subprocess.Popen(cmd, shell=True)
        logger.info("MLflow server started successfully")
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to start MLflow server: {str(e)}")
        raise

def start_datapipeline():
    """
    Start the data pipeline process using DVC experiment.
    Handles termination of any existing pipeline process before starting new one.
    
    Raises:
        subprocess.SubprocessError: If pipeline process fails to start or terminate.
    """
    global pipeline_process
    try:
        # If an old process exists, terminate it
        if pipeline_process and pipeline_process.poll() is None:
            logger.info("Terminating previous data pipeline...")
            pipeline_process.terminate()
            try:
                pipeline_process.wait(timeout=5)
                logger.info("Previous data pipeline terminated successfully")
            except subprocess.TimeoutExpired:
                logger.warning("Force killing the previous process...")
                pipeline_process.kill()
                pipeline_process.wait()
                logger.info("Previous process force killed")

        logger.info("Starting new data pipeline...")
        cmd = ["dvc", "exp", "run"]
        pipeline_process = subprocess.Popen(cmd)
        logger.info("Data pipeline started successfully")
        
    except subprocess.SubprocessError as e:
        logger.error(f"Data pipeline process error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Update params.yaml with current date
    with open(file_path,"r") as f:
        data = yaml.load(f)
    data["data"]["date"] = str(date.today())
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    # Start in dedicated thread
    run_mlflow_server()
    time.sleep(5)  # Give server time to start

    # Start the data pipeline in a separate thread
    while(True):
        data = pd.read_csv("data/news_feed.csv")
        if len(data) > 3:
            start_datapipeline()
        time.sleep(600)