import os
import zipfile

# CONFIGURATION
OUTPUT_FILENAME = "phoenix_deploy.zip"
MAX_FILE_SIZE_MB = 50  # Skip any single file bigger than 50MB
FOLDERS_TO_INCLUDE = ["app", "resources"]
FILES_TO_INCLUDE = ["run_full_video.py", ".env"]

# IGNORE LIST (Folders/Files to strictly skip)
IGNORE_FOLDERS = [
    "__pycache__", "venv", "env", ".git", ".vscode", 
    "zeroscope_cache", "model_cache", "render_output", 
    "wandb", "tmp", "node_modules"
]
IGNORE_EXTENSIONS = [".mp4", ".avi", ".mov", ".zip", ".7z", ".rar"]

def pack():
    print(f"📦 STARTING SMART PACK: {OUTPUT_FILENAME}")
    base_dir = os.getcwd()
    
    with zipfile.ZipFile(OUTPUT_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add Root Files
        for fname in FILES_TO_INCLUDE:
            if os.path.exists(fname):
                print(f"  + Added: {fname}")
                zipf.write(fname)
            else:
                print(f"  ⚠️ WARNING: Could not find {fname}")

        # 2. Add Folders (Recursive)
        for folder in FOLDERS_TO_INCLUDE:
            if not os.path.exists(folder):
                print(f"  ⚠️ WARNING: Could not find folder {folder}")
                continue
                
            for root, dirs, files in os.walk(folder):
                # Remove ignored folders in-place so we don't walk them
                dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]
                
                for file in files:
                    # Check extension
                    if any(file.endswith(ext) for ext in IGNORE_EXTENSIONS):
                        continue
                        
                    file_path = os.path.join(root, file)
                    
                    # Check Size
                    try:
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        if size_mb > MAX_FILE_SIZE_MB:
                            print(f"  🔴 SKIPPED (Too Big): {file_path} ({size_mb:.2f} MB)")
                            continue
                    except: pass

                    # Add to Zip
                    print(f"  + Added: {file_path}")
                    zipf.write(file_path)

    print(f"\n✅ SUCCESS: Created {OUTPUT_FILENAME}")
    print(f"📊 Size: {os.path.getsize(OUTPUT_FILENAME) / (1024*1024):.2f} MB")
    print("👉 Upload THIS file to Colab.")

if __name__ == "__main__":
    pack()