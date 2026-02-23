import os, sys, shutil
from pathlib import Path

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

repo_id = "hssling/derm-analyzer-api"

import subprocess

# Clone the existing HF Space
result = subprocess.run(
    ["git", "clone", f"https://hssling:{token}@huggingface.co/spaces/{repo_id}", "/tmp/hf_space"],
    capture_output=True, text=True
)
print("Clone stdout:", result.stdout)
print("Clone stderr:", result.stderr)

# Ensure we use a clean .gitattributes with NO .py LFS tracking 
# (HF default tracks *.py as non-LFS text, our override was causing conflicts)
with open("/tmp/hf_space/.gitattributes", "w") as f:
    f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
    f.write("*.ckpt filter=lfs diff=lfs merge=lfs -text\n")

# Copy only Space-needed files
for fname in ["app.py", "requirements.txt", "README.md"]:
    src = Path(fname)
    if src.exists():
        shutil.copy(src, f"/tmp/hf_space/{fname}")
        print(f"Copied {fname}")

# Commit and push
os.chdir("/tmp/hf_space")
subprocess.run(["git", "config", "user.email", "ci@github.com"])
subprocess.run(["git", "config", "user.name", "GitHub Actions"])
subprocess.run(["git", "add", "."])
result = subprocess.run(["git", "diff", "--staged", "--stat"], capture_output=True, text=True)
print("Changes:", result.stdout)
subprocess.run(["git", "commit", "-m", "Sync from GitHub CI", "--allow-empty"])
result = subprocess.run(
    ["git", "push"],
    capture_output=True, text=True
)
print("Push stdout:", result.stdout)
print("Push stderr:", result.stderr)
if result.returncode != 0:
    print(f"Push failed with exit code {result.returncode}")
    sys.exit(result.returncode)
print(f"Successfully synced to https://huggingface.co/spaces/{repo_id}")

