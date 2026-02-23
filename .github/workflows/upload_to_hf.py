import os, sys, shutil
from huggingface_hub import upload_folder
from pathlib import Path

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

repo_id = "hssling/derm-analyzer-api"

# Copy only Space-needed files into a temp folder
tmp = Path("/tmp/hf_space")
tmp.mkdir(exist_ok=True)
for fname in ["app.py", "requirements.txt", "README.md"]:
    src = Path(fname)
    if src.exists():
        shutil.copy(src, tmp / fname)
        print(f"Copied {fname}")

upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=str(tmp),
    token=token,
    commit_message="Sync from GitHub CI",
    ignore_patterns=["*.pyc", "__pycache__"],
)
print(f"Successfully uploaded to https://huggingface.co/spaces/{repo_id}")
