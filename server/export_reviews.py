"""
Packs the saved recordings and the team's reviews into one file, to send for tuning (run by export_reviews.bat on
the team's computer, inside the server's container, which already holds HF_TOKEN and FASEEH_DATASET).

  python export_reviews.py [--no-audio]     ->  /tmp/reviews.zip
     labels.json   the team's verdicts (review.html)
     meta/*.json   what the model said about each recording
     audio/*       the recordings themselves (left out with --no-audio)
"""
import json
import os
import sys
import zipfile

dataset, token = os.environ.get("FASEEH_DATASET", "").strip(), os.environ.get("HF_TOKEN", "").strip()
if not dataset or not token:
    sys.exit("HF_TOKEN / FASEEH_DATASET are not set in settings.env: nothing to export.")
from huggingface_hub import snapshot_download

want = ["labels.json", "meta/*"] + ([] if "--no-audio" in sys.argv else ["audio/*"])
out = "/tmp/reviews"
snapshot_download(dataset, repo_type="dataset", token=token, local_dir=out, allow_patterns=want)
n = {"meta": 0, "audio": 0}
with zipfile.ZipFile("/tmp/reviews.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for folder, _, files in os.walk(out):
        if ".cache" in folder.split(os.sep):
            continue
        for f in files:
            path = os.path.join(folder, f)
            rel = os.path.relpath(path, out)
            z.write(path, rel)
            top = rel.split(os.sep)[0]
            if top in n:
                n[top] += 1
try:
    labels = len(json.load(open(os.path.join(out, "labels.json"))))
except Exception:
    labels = 0
print(f"recordings: {n['meta']}, with audio: {n['audio']}, reviewed: {labels}")
print(f"/tmp/reviews.zip: {os.path.getsize('/tmp/reviews.zip') / 1e6:.1f} MB")
