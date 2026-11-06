"""
Keeps the Faseeh analysis server on this computer running the latest version from GitHub.

Runs inside the container (see Dockerfile / boot.sh). The repository is the only source:
  - the server code, its settings (config.json) and calibration (calibration.json) come from server/hf_space,
    the same files the Hugging Face Space runs, so both servers always give the same results;
  - every CHECK_SECONDS it looks for a new version on GitHub; when the server files changed it installs any
    new requirements and restarts the server (about half a minute while the model loads again);
  - when this file itself changed it stops, and boot.sh starts the new version;
  - if the server stops for any reason it is started again.
The model is downloaded by the server the first time and kept in the cache volume; a different model name
in the code is downloaded by itself.
"""
import hashlib
import os
import subprocess
import sys
import time

REPO = os.environ.get("FASEEH_SRC", "/src/repo")
BRANCH = os.environ.get("FASEEH_BRANCH", "main")
APP = os.path.join(REPO, "server", "hf_space")
WATCH = [APP, os.path.join(REPO, "server", "local")]
ME = os.path.abspath(__file__)
CHECK_SECONDS = int(os.environ.get("FASEEH_CHECK_SECONDS", "120"))
STAMP = os.path.expanduser("~/.requirements.sha")   # in the container itself: a new container checks again


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def digest(paths):
    h = hashlib.sha256()
    for root in paths:
        if os.path.isfile(root):
            h.update(open(root, "rb").read())
            continue
        for d, dirs, files in sorted(os.walk(root)):
            dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", "audio_cache"))
            for f in sorted(files):
                p = os.path.join(d, f)
                h.update(p.encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def pull():
    """Brings the newest version from GitHub. False when GitHub cannot be reached (the current copy keeps running)."""
    r = subprocess.run(["git", "-C", REPO, "fetch", "--depth", "1", "origin", BRANCH], capture_output=True, text=True)
    if r.returncode:
        log("GitHub not reachable, keeping the current version: " + r.stderr.strip()[-200:])
        return False
    subprocess.run(["git", "-C", REPO, "reset", "--hard", "FETCH_HEAD"], capture_output=True)
    return True


def install_requirements():
    req = os.path.join(APP, "requirements.txt")
    if not os.path.exists(req):
        return
    want = digest([req])
    if os.path.exists(STAMP) and open(STAMP).read() == want:
        return
    log("installing the server's requirements ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", req])
    if r.returncode == 0:
        open(STAMP, "w").write(want)


def start():
    log("starting the server (the model loads in about half a minute; the first time it is downloaded, ~2.5 GB)")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "faseeh_server:app", "--host", "0.0.0.0", "--port", "7860"],
                            cwd=APP)


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    import torch
    log("graphics card: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT FOUND - the server runs on the processor (slow)"))
    me, files = digest([ME]), digest(WATCH)
    install_requirements()
    server = start()
    while True:
        time.sleep(CHECK_SECONDS)
        if server.poll() is not None:
            log(f"the server stopped (code {server.returncode}), starting it again")
            server = start()
        if not pull():
            continue
        if digest([ME]) != me:
            log("a new version of the supervisor: restarting everything")
            stop(server)
            sys.exit(0)
        now = digest(WATCH)
        if now != files:
            files = now
            log("new server version from GitHub")
            install_requirements()
            stop(server)
            server = start()


if __name__ == "__main__":
    main()
