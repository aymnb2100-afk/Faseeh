"""
Recordings that readers chose to share, kept to improve Faseeh, and the team's review of them.

Saving
  A recording is kept only when the page sends X-Save: 1 (the reader ticked "help improve Faseeh"; the team's
  own recordings always). It is stored, with what the model found, in a PRIVATE Hugging Face dataset:
      audio/<id>.<ext>   the recording exactly as it arrived
      meta/<id>.json     when, surah and ayahs, the model's result, who and what for (team recordings only)
      labels.json        the team's review of every recording, one entry per id
  Every kept recording says whose it is, so one reader's recordings can be told from another's: the name the
  reader typed, or, if they chose to stay anonymous, a made-up name fixed to their browser ("مجهول 4K7Q").
  Nothing else about them is kept (no address or device). Only the team (with the key) can set the purpose
  ("tune": used to improve the settings, "test": kept apart to measure them), so the two never mix. Saving happens in the background, in batches, and can never slow down or break an analysis.

Review (team only, the X-Team-Key header must match FASEEH_TEAM_KEY; wrong keys are limited per minute)
  GET  /review/check            is this team key right? (the page checks it before saving team recordings)
  GET  /review/list             every recording with its meta and label
  GET  /review/audio/<id>       the recording itself
  POST /review/label            {"id", "verdict", "mistakes": [[ayah, word], ...], "note"}
     verdict: "model_right" (the model's answer is the truth), "model_wrong" (mistakes = the real ones),
              "silent", "invalid" (unusable)

Settings (environment / Space secrets):  FASEEH_DATASET  e.g. LowkeyO/faseeh-recordings
                                          HF_TOKEN        write access to that dataset
                                          FASEEH_TEAM_KEY the team's password for the review page
                                          FASEEH_SERVER_NAME which server analysed it (hf / local)
"""
import collections
import datetime
import hmac
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

DATASET = os.environ.get("FASEEH_DATASET", "").strip()
TOKEN = os.environ.get("HF_TOKEN", "").strip()
TEAM_KEY = os.environ.get("FASEEH_TEAM_KEY", "").strip()
SERVER_NAME = os.environ.get("FASEEH_SERVER_NAME", "hf").strip() or "hf"
ENABLED = bool(DATASET and TOKEN)
BATCH_SECONDS, BATCH_MAX, PENDING_MAX = 20, 20, 300
VERDICTS = ("model_right", "model_wrong", "silent", "invalid")
# Limits that keep a flood of requests from filling the dataset or guessing the key (generous for real use:
# a whole class recording from one network stays far below them).
SAVES_PER_HOUR = int(os.environ.get("FASEEH_SAVES_PER_HOUR", "1500"))            # all readers together
SAVES_PER_READER = int(os.environ.get("FASEEH_SAVES_PER_READER", "400"))         # one network address, per hour
WRONG_KEYS_PER_MINUTE = 20

_api = None
_queue: "queue.Queue" = queue.Queue()
_pending: dict = {}                       # id -> meta, saved but not yet committed (shown in the review list too)
_meta_cache: dict = {}                    # id -> meta (never changes once written)
_labels_lock = threading.Lock()


def log(msg):
    print(time.strftime("%H:%M:%S"), "[recordings]", msg, flush=True)


def api():
    global _api
    if _api is None:
        from huggingface_hub import HfApi
        _api = HfApi(token=TOKEN)
    return _api


def status() -> dict:
    return {"saving": ENABLED, "review": ENABLED and bool(TEAM_KEY)}


# ------------------------------------------------------------------------------------------- saving
def _same_key(given: str) -> bool:
    """Compared in a way that takes the same time whether the first or the last character is wrong."""
    return bool(TEAM_KEY) and hmac.compare_digest((given or "").encode(), TEAM_KEY.encode())


def _is_team(request: Request) -> bool:
    return _same_key(request.headers.get("x-team-key", ""))


_saved_at: "collections.deque" = collections.deque()      # (time, reader) of the recordings kept in the last hour


def _reader(request: Request) -> str:
    """Where the request came from, as the tunnel in front of the server reports it (only used to count)."""
    first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return first or (request.client.host if request.client else "")


def _may_save(request: Request) -> bool:
    now, who = time.time(), _reader(request)
    while _saved_at and now - _saved_at[0][0] > 3600:
        _saved_at.popleft()
    if len(_saved_at) >= SAVES_PER_HOUR or sum(1 for _, w in _saved_at if w == who) >= SAVES_PER_READER:
        return False
    _saved_at.append((now, who))
    return True


def _extension(content_type: str) -> str:
    ct = (content_type or "").lower()
    for key, ext in (("webm", "webm"), ("ogg", "ogg"), ("mp4", "m4a"), ("m4a", "m4a"), ("aac", "m4a"),
                     ("mpeg", "mp3"), ("wav", "wav")):
        if key in ct:
            return ext
    return "bin"


def save(request: Request, body: bytes, ayahs: list, result: dict | None = None, seconds: float = 0.0,
         auto: str | None = None):
    """Queue one shared recording. Never raises."""
    try:
        if not ENABLED or request.headers.get("x-save") != "1" or not body:
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        rid = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        team = _is_team(request)
        speaker = re.sub(r"[\x00-\x1f\x7f<>]", "", urllib.parse.unquote(request.headers.get("x-speaker", "")))[:40].strip()
        purpose = request.headers.get("x-purpose", "tune") if team else "tune"
        if purpose not in ("tune", "test"):
            purpose = "tune"
        errors = []
        for key, items in (result or {}).items():
            if key.startswith("_"):
                continue
            for e in items:
                errors.append({"ayah": int(key.split(":")[1]), "word": int(e["positionIndex"]) + 1,
                               "letter": e.get("letter", ""), "type": e.get("errorType", "")})
        meta = {
            "id": rid, "time": now.isoformat(timespec="seconds"), "team": team, "speaker": speaker,
            "anonymous": request.headers.get("x-anonymous") == "1", "purpose": purpose,
            "surah": ayahs[0][0], "ayahs": [a for _, a in ayahs], "seconds": round(seconds, 2),
            "audio": f"audio/{rid}.{_extension(request.headers.get('content-type', ''))}",
            "server": SERVER_NAME, "auto": auto, "mismatch": bool((result or {}).get("_mismatch")),
            "model_errors": errors, "settings": {k: v for k, v in ((result or {}).get("_meta") or {}).items()
                                                 if k in ("votes", "minConf", "calibration", "calibMode", "device")},
        }
        if len(_pending) >= PENDING_MAX:
            log("too many recordings waiting to be saved, this one is skipped")
            return
        if not _may_save(request):
            log("too many recordings kept in the last hour, this one is skipped")
            return
        _pending[rid] = meta
        _queue.put((rid, body, meta))
    except Exception as e:                                   # saving must never break an analysis
        log(f"not saved: {e}")


def _worker():
    from huggingface_hub import CommitOperationAdd
    batch = []
    while True:
        try:
            item = _queue.get(timeout=BATCH_SECONDS)
            batch.append(item)
            if len(batch) < BATCH_MAX:
                continue
        except queue.Empty:
            pass
        if not batch:
            continue
        ops = []
        for rid, body, meta in batch:
            ops.append(CommitOperationAdd(path_in_repo=meta["audio"], path_or_fileobj=body))
            ops.append(CommitOperationAdd(path_in_repo=f"meta/{rid}.json",
                                          path_or_fileobj=json.dumps(meta, ensure_ascii=False, indent=1).encode()))
        try:
            api().create_commit(repo_id=DATASET, repo_type="dataset", operations=ops,
                                commit_message=f"{len(batch)} recording(s) from {SERVER_NAME}")
            for rid, _, meta in batch:
                _meta_cache[rid] = meta
                _pending.pop(rid, None)
            log(f"saved {len(batch)} recording(s)")
            batch = []
        except Exception as e:                               # keep them and try again with the next batch
            log(f"saving failed, will retry: {e}")
            time.sleep(30)


if ENABLED:
    threading.Thread(target=_worker, daemon=True).start()
    log(f"sharing on: recordings go to the private dataset {DATASET}")


# ------------------------------------------------------------------------------------------- review
_wrong_keys: "collections.deque" = collections.deque()    # times of the wrong keys tried in the last minute


def _require_team(request: Request, key: str | None = None):
    if not ENABLED or not TEAM_KEY:
        raise HTTPException(503, "Recording review is not set up on this server.")
    now = time.time()
    while _wrong_keys and now - _wrong_keys[0] > 60:
        _wrong_keys.popleft()
    if len(_wrong_keys) >= WRONG_KEYS_PER_MINUTE:            # guessing the key is slowed to a crawl
        raise HTTPException(429, "Too many wrong keys. Wait a minute.")
    if not _same_key(key or request.headers.get("x-team-key", "")):
        _wrong_keys.append(now)
        raise HTTPException(403, "Wrong team key.")


def _download(path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATASET, path, repo_type="dataset", token=TOKEN)


def _read_labels() -> dict:
    try:
        with open(_download("labels.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                                   # no review saved yet: the file does not exist
        if "EntryNotFound" in type(e).__name__ or "404" in str(e):
            return {}
        raise


def list_recordings(request: Request) -> dict:
    _require_team(request)
    files = api().list_repo_files(DATASET, repo_type="dataset")
    for path in files:
        if path.startswith("meta/") and path.endswith(".json"):
            rid = path[5:-5]
            if rid not in _meta_cache:
                try:
                    with open(_download(path), encoding="utf-8") as f:
                        _meta_cache[rid] = json.load(f)
                except Exception as e:
                    log(f"could not read {path}: {e}")
    labels = _read_labels() if "labels.json" in files else {}
    items = [dict(m, saved=True, label=labels.get(rid)) for rid, m in _meta_cache.items()]
    items += [dict(m, saved=False, label=None) for rid, m in list(_pending.items()) if rid not in _meta_cache]
    items.sort(key=lambda m: m["id"], reverse=True)
    return {"recordings": items}


def audio_file(request: Request, rid: str, key: str):
    _require_team(request, key)
    meta = _meta_cache.get(rid)
    if not meta:
        raise HTTPException(404, "Unknown recording (open the list first).")
    ext = meta["audio"].rsplit(".", 1)[-1]
    kinds = {"webm": "audio/webm", "ogg": "audio/ogg", "m4a": "audio/mp4", "mp3": "audio/mpeg", "wav": "audio/wav"}
    return FileResponse(_download(meta["audio"]), media_type=kinds.get(ext, "application/octet-stream"))


async def set_label(request: Request) -> dict:
    _require_team(request)
    data = await request.json()
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_write_label, data)


def _write_label(data: dict) -> dict:
    rid, verdict = str(data.get("id", "")), data.get("verdict")
    if verdict not in VERDICTS or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", rid):
        raise HTTPException(400, "id and a verdict are required.")
    try:
        mistakes = sorted({(int(a), int(w)) for a, w in data.get("mistakes") or []})[:400]
    except (TypeError, ValueError):
        raise HTTPException(400, "mistakes must be [[ayah, word], ...].")
    label = {"verdict": verdict, "mistakes": [list(m) for m in mistakes],
             "note": str(data.get("note", ""))[:300], "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with _labels_lock:                                       # read, change one entry, write back
        labels = _read_labels()
        labels[rid] = label
        api().upload_file(path_or_fileobj=json.dumps(labels, ensure_ascii=False, indent=1).encode(),
                          path_in_repo="labels.json", repo_id=DATASET, repo_type="dataset",
                          commit_message=f"review {rid}: {verdict}")
    return {"ok": True, "label": label}


def attach(app):
    """Adds the review endpoints to the server (plain functions run in a worker thread, so the
    downloads from the dataset never hold up an analysis)."""
    def review_list(request: Request):
        return list_recordings(request)

    async def review_label(request: Request):
        return await set_label(request)

    def review_audio(request: Request, rid: str, key: str = ""):
        return audio_file(request, rid, key)

    def review_check(request: Request):
        _require_team(request)
        return {"ok": True}

    app.add_api_route("/review/check", review_check, methods=["GET"])
    app.add_api_route("/review/list", review_list, methods=["GET"])
    app.add_api_route("/review/label", review_label, methods=["POST"])
    app.add_api_route("/review/audio/{rid}", review_audio, methods=["GET"])
