import subprocess
import atexit
import time
from threading import Thread
import pandas as pd

pipeline_process = None

def run_mlflow_server():
    """Thread target function for server"""
    cmd = ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
    process = subprocess.Popen(cmd)
    atexit.register(process.terminate)
    process.wait()  # Blocks until server terminates

def start_datapipeline():
    """Start the data pipeline process"""
    global pipeline_process
    # If an old process exists, terminate it
    if pipeline_process and pipeline_process.poll() is None:
        print("Terminating previous data pipeline...")
        pipeline_process.terminate()
        try:
            pipeline_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing the process...")
            pipeline_process.kill()

    print("Starting new data pipeline...")
    cmd = ["dvc", "exp", "run"]
    pipeline_process = subprocess.Popen(cmd)


if __name__ == "__main__":
    # Start in dedicated thread
    server_thread = Thread(target=run_mlflow_server, daemon=True)
    server_thread.start()
    time.sleep(5)  # Give server time to start

    # Start the data pipeline in a separate thread
    while(True):
        data = pd.read_csv("news_feed.csv")
        if len(data) > 10:
            start_datapipeline()
        time.sleep(600)