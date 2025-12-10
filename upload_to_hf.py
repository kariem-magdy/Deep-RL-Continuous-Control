from huggingface_hub import HfApi
import argparse

def upload(model_path, repo_id, model_name, env_name):
    api = HfApi()
    
    print(f"Creating/Checking Repo: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    print(f"Uploading {model_path}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"{model_name}_{env_name}.pth",
        repo_id=repo_id,
        commit_message=f"Upload {model_name} for {env_name}"
    )
    
    readme = f"""
---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- {env_name}
library_name: pytorch
---
# {model_name} Agent for {env_name}
Trained with PyTorch.
    """
    with open("README.md", "w") as f:
        f.write(readme)
        
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id
    )
    print("Upload Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    args = parser.parse_args()
    
    upload(args.path, args.repo, args.model, args.env)