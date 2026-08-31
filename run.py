"""Run the AquaMind backend and static frontend together."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
BACKEND_URL = "http://127.0.0.1:5000/api/health"
FRONTEND_URL = "http://127.0.0.1:8080/login.html"


def wait_for_backend(timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_URL, timeout=1):
                return True
        except Exception:
            time.sleep(0.25)
    return False


def main() -> int:
    processes = [
        subprocess.Popen([sys.executable, "main.py"], cwd=BACKEND),
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "8080", "--bind", "127.0.0.1"],
            cwd=FRONTEND,
        ),
    ]
    try:
        if not wait_for_backend():
            print("Backend did not become ready within 30 seconds.", file=sys.stderr)
            return 1
        webbrowser.open(FRONTEND_URL)
        print(f"AquaMind frontend: {FRONTEND_URL}")
        print(f"AquaMind backend:  http://127.0.0.1:5000")
        print("Press Ctrl+C to stop both services.")
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return next((process.returncode for process in processes if process.returncode), 0)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
