from huggingface_hub import HfFolder
import os

token = os.environ.get("HF_TOKEN")
HfFolder.save_token(token)
print("Logged in successfully")
