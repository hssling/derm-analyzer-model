import os, sys, traceback
from huggingface_hub import HfApi

def deploy():
    try:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: HF_TOKEN not set")
            sys.exit(1)

        repo_id = "hssling/derm-analyzer-api"
        api = HfApi(token=token)

        # Upload files one by one to see which one fails
        files = ["app.py", "requirements.txt", "README.md"]
        for f in files:
            if os.path.exists(f):
                print(f"Uploading {f}...")
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=f,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"Successfully uploaded {f}")
            else:
                print(f"File {f} not found, skipping.")
        
        print("Deployment complete!")
    except Exception as e:
        print("DEPLOYMENT FAILED!")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    deploy()
