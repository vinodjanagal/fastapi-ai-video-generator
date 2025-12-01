# --- kaggle_bridge_clean.py ---
# Run this as the first cell in a Kaggle Notebook (GPU enabled, Internet ON).
# It installs Python deps (no system apt), downloads the Linux x64 VS Code CLI,
# and starts the VS Code Remote Tunnel in a streaming mode so you can authenticate.

import os
import subprocess
import time
import sys
from pathlib import Path

TUNNEL_NAME = "phoenix-gpu"   # name that will appear in VS Code Remote Tunnels
VSCODE_CLI_TAR = "vscode_cli.tar.gz"
VSCODE_DIR = "code"

def run(cmd, check=False, env=None):
    print(f"> {cmd}")
    return subprocess.run(cmd, shell=True, check=check, env=env)

def install_python_deps():
    print("📦 Installing Python dependencies (diffusers, accelerate, transformers, moviepy...)")
    # This is the heavy pip line. On Kaggle this will use the GPU environment and cached wheels.
    pip_cmd = (
        "pip install -q "
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 "
        "diffusers transformers accelerate "
        "moviepy imageio numpy scipy ftfy safetensors"
    )
    run(pip_cmd, check=True)
    print("✅ Python deps installed.")

def download_vscode_cli():
    # Downloads the Linux x64 CLI which works on Kaggle (Debian/Ubuntu-like)
    if os.path.exists(VSCODE_DIR) and os.path.isdir(VSCODE_DIR):
        print("✅ VS Code CLI already present.")
        return

    print("⬇️ Downloading VS Code CLI (linux x64)...")
    run(f"curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64' --output {VSCODE_CLI_TAR}", check=True)
    run(f"tar -xf {VSCODE_CLI_TAR}", check=True)
    # extracted 'code' binary directory appears; ensure it's executable
    if not os.path.exists("./code"):
        raise RuntimeError("Expected './code' after extracting vscode_cli.tar.gz, but it was not found.")
    run("chmod +x ./code && mv ./code ./code_cli || true")
    # make the binary accessible in this working dir
    print("✅ VS Code CLI ready (./code_cli).")

def start_tunnel():
    print("\n🔗 Starting VS Code Remote Tunnel (keep this cell running) ...")
    print("IMPORTANT: You will see a GitHub Device Code in the logs. Follow the instructions printed below.")
    print("1) Open https://github.com/login/device in your browser")
    print("2) Enter the code reported in the tunnel logs")
    print("3) In local VS Code: install 'Remote - Tunnels' extension, open Remote Explorer -> Tunnels -> connect to 'phoenix-gpu'\n")

    # We use Popen so we can stream stdout/stderr lines and keep the notebook cell alive.
    try:
        # Use the extracted binary (named code_cli). If you have a different name, adjust accordingly.
        cmd = "./code_cli tunnel --name {} --accept-server-license-terms".format(TUNNEL_NAME)
        print(f"> Starting: {cmd}\n")
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream and show the output so you can grab the GitHub device code.
        for line in process.stdout:
            # Print each line as it arrives (keeps the Kaggle cell alive)
            print(line, end="")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n🛑 Tunnel stopped by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\n❌ Tunnel process failed: {e}")

if __name__ == "__main__":
    # Safety: run from /kaggle/working if possible
    print("Working dir:", os.getcwd())
    # 1) Install deps
    try:
        install_python_deps()
    except Exception as e:
        print("⚠️ Python deps installation failed:", e)
        print("You can still try to continue if core packages already exist.")
    # 2) Download VS Code CLI (linux x64)
    try:
        download_vscode_cli()
    except Exception as e:
        print("❌ VS Code CLI download/extract failed:", e)
        raise

    # 3) Start interactive tunnel (this blocks and streams logs)
    start_tunnel()
