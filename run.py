"""
run.py
======
Main launcher script for the Oral Cancer Screening App.

Starts uvicorn on port 8000 and automatically opens http://127.0.0.1:8000/screening
in the browser once the server is ready.
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import uvicorn


def stop_existing_server(port: int = 8000) -> None:
    """Stop any old server on the target port if still running."""
    if sys.platform != "win32":
        return
    try:
        cmd = f'netstat -aon | findstr :{port}'
        output = subprocess.check_output(cmd, shell=True).decode(errors="ignore")
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"Stopping lingering process {pid} on port {port}...")
                    subprocess.run(
                        f"taskkill /F /PID {pid}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
    except Exception:
        pass


def open_browser_when_ready(url: str, port: int = 8000, timeout: float = 10.0) -> None:
    """Wait until the FastAPI server responds, then open the browser."""
    start_time = time.time()
    health_url = f"http://127.0.0.1:{port}/api/health"
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    print(f"\nServer ready! Opening {url} in your browser...")
                    webbrowser.open(url)
                    return
        except Exception:
            pass
        time.sleep(0.3)
    # If polling times out, open the URL as fallback
    webbrowser.open(url)


if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    screening_url = f"http://{host}:{port}/screening"

    # Stop any old server on port 8000
    stop_existing_server(port)

    print("\n" + "=" * 65)
    print("  Oral Pre-Cancer Screening Application")
    print("  --------------------------------------")
    print(f"  Web Interface:      {screening_url}")
    print(f"  Home / Model Info:  http://{host}:{port}/about")
    print(f"  API Docs (Swagger): http://{host}:{port}/docs")
    print("=" * 65 + "\n")

    # Start browser opener in a background thread
    threading.Thread(
        target=open_browser_when_ready,
        args=(screening_url, port),
        daemon=True,
    ).start()

    # Start uvicorn server on port 8000
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
    )
