import os, sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

repo_id = "hssling/derm-analyzer-api"
api = HfApi()

files_to_upload = ["app.py", "requirements.txt", "README.md"]

for file in files_to_upload:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space",
            token=token
        )
        print(f"Uploaded {file}")
    else:
        print(f"Warning: {file} not found")

print(f"Successfully synced to https://huggingface.co/spaces/{repo_id}")
