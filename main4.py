"""
main4.py
Single entry point for AREIS. Launches llama-server.exe (CUDA build)
silently in the background, waits until it's ready, then opens the GUI.
No browser, no visible terminal, no network access required -- the server
only listens on 127.0.0.1 (localhost), which never leaves this machine.
"""

import os
import sys
import time
import subprocess
import requests

from gui4 import launch_app

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LLAMA_EXE = os.path.join(BASE_DIR, "cuda", "llama-b10853-bin-win-cuda-12.4-x64", "llama-server.exe")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3.5-9B-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(BASE_DIR, "models", "mmproj-F16.gguf")
HEALTH_URL = "http://127.0.0.1:8080/health"

llama_process = None


def start_llama_server(timeout_seconds=90):
    global llama_process

    for path, label in ((LLAMA_EXE, "llama-server.exe"),
                         (MODEL_PATH, "main model"),
                         (MMPROJ_PATH, "mmproj vision file")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    llama_process = subprocess.Popen(
        [
            LLAMA_EXE,
            "-m", MODEL_PATH,
            "--mmproj", MMPROJ_PATH,
            "-ngl", "99",
            "-c", "1024",
            "--port", "8080",
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the server responds before handing off to the GUI
    waited = 0
    while waited < timeout_seconds:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        waited += 1

    raise RuntimeError(
        "llama-server.exe did not become ready in time. "
        "Check that the CUDA build and model files are in the right folders."
    )


def stop_llama_server():
    if llama_process is not None:
        llama_process.terminate()


if __name__ == "__main__":
    print("Starting AREIS local model server, this may take a moment...")
    try:
        start_llama_server()
    except Exception as e:
        print(f"Failed to start model server: {e}")
        sys.exit(1)

    try:
        launch_app()
    finally:
        stop_llama_server()