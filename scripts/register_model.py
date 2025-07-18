import subprocess
import os

def register_model():
    script_path = os.path.join(os.path.dirname(__file__), 'model_registry_final_new.py')
    subprocess.run(["python", script_path], check=True)