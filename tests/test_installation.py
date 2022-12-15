import pytest
import sys
import os


not_
platform: str = sys.platform

if platform not in ["linux", "linux2"]:
    print(f"Error: {not_lin_err}")

os.system("pip install --quiet .")
os.system("rm -r dLux.egg-info")
os.system("rm -r build")

