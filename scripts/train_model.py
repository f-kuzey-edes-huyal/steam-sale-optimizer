import subprocess
import os

def train_model():
    script_path = os.path.join(os.path.dirname(__file__), 'train_last_new.py')
    subprocess.run(["python", script_path], check=True)