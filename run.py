"""Launch script for Visual Studio — run this file to start the API."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run([sys.executable, "-m", "uvicorn", "cadsentinel.api.main:app", "--reload"])