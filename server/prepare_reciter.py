"""
Prepares the new reference reciter, Muhammad Siddiq al-Minshawi (murattal), on the team's computer.

Runs inside the local server's container, on its graphics card: start the server (start_faseeh.bat), then
prepare_reciter.bat. It takes about half an hour and leaves one file to send back: reciter_out.zip, with
  audio/<surah>.mp3       his recordings of the website's surahs (quran.com), and audio_urls.json
  align_data.npz          what the model hears in them: word timings and his madd times are made from it
                          (align_reciter.py --data, build_madd_reference.py)
  calibration.json        the model's habitual slips on his correct reading, calibrated on him alone
  test_data.npz           what the model hears in every ayah read by the test reciters (ash-Shatri, Maher), to
                          measure the half-second madd rule before it goes live
  results/summary.json    the evaluation on the test reciters with the new calibration (and the old, to compare)
Nothing on the running server changes.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FASEEH_OUT", "/tmp/reciter_out")
sys.path[:0] = [os.path.join(HERE, "hf_space"), HERE]
os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
os.environ["FASEEH_RESULTS"] = os.path.join(OUT, "results")

import numpy as np

import faseeh_server as fs
import madd_timing as mt
from quran_transcript import Aya, quran_phonetizer

SURAHS = {78: 40, 80: 42, 103: 3, 108: 3, 112: 4, 113: 5, 114: 6}
EVERYAYAH = "Minshawy_Murattal_128kbps"          # the same reciter, one file per ayah (calibration)
TEST_RECITERS = ["Abu_Bakr_Ash-Shaatree_128kbps", "MaherAlMuaiqly128kbps"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Faseeh"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=180).read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def mapping(m):
    if m is None:
        return None
    pos = getattr(m, "pos", None)
    return [int(pos[0]), int(pos[1]), bool(getattr(m, "deleted", False))] if pos is not None else None


def ayah_info(s, a):
    text = Aya(sura_idx=s, aya_idx=a).get().uthmani
    ref = quran_phonetizer(text, fs.MOSHAF, remove_spaces=True, sura_idx=s)
    return {"text": text, "phonemes": list(ref.phonemes), "mappings": [mapping(m) for m in ref.mappings]}


# ------------------------------------------------------------------------- 1. his recordings (quran.com)
def minshawi_urls():
    """quran.com's murattal recitation by al-Minshawi, found by name (its number there is 9)."""
    rid = 9
    try:
        recs = json.loads(get("https://api.quran.com/api/v4/resources/recitations?language=en"))["recitations"]
        for r in recs:
            if "minshawi" in (r.get("reciter_name") or "").lower() and (r.get("style") or "").lower() == "murattal":
                rid = r["id"]
                log(f'quran.com recitation {rid}: {r.get("reciter_name")} ({r.get("style")})')
                break
    except Exception as e:
        log(f"quran.com list not reachable ({e}), using recitation {rid}")
    urls = {}
    for s in SURAHS:
        data = json.loads(get(f"https://api.quran.com/api/v4/chapter_recitations/{rid}/{s}"))
        urls[s] = data["audio_file"]["audio_url"]
    return rid, urls


# ------------------------------------------------------------------------- 2. what the model hears in them
def his_recordings():
    rid, urls = minshawi_urls()
    json.dump({"recitation": rid, "urls": urls}, open(os.path.join(OUT, "audio_urls.json"), "w"), indent=1)
    tok = fs.MODEL.multi_level_tokenizer.get_tokenizer()
    meta = {"vocab": tok.get_vocab(), "blank": tok.pad_token_id, "sr": fs.SR, "surahs": {}, "reciter": "minshawi"}
    arrays = {}
    for s in SURAHS:
        log(f"surah {s}: downloading {urls[s]}")
        raw = get(urls[s])
        open(os.path.join(OUT, "audio", f"{s}.mp3"), "wb").write(raw)
        wave = fs.to_wave(raw)
        log(f"surah {s}: {len(wave) / fs.SR:.0f} s, listening ...")
        lp, fps = mt.emissions(fs.MODEL, wave, fs.SR)
        arrays[f"lp{s}"] = lp.astype(np.float16)
        meta["surahs"][s] = {"fps": fps, "seconds": len(wave) / fs.SR,
                             "ayahs": {a: ayah_info(s, a) for a in range(1, SURAHS[s] + 1)}}
    np.savez_compressed(os.path.join(OUT, "align_data.npz"), meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    log("align_data.npz saved")


# ------------------------------------------------------------------------- 3. calibration on him alone
def calibrate():
    ayahs = [(s, a) for s, n in SURAHS.items() for a in range(1, n + 1)] + [(5, 109)]
    slips, done = {}, 0
    for s, a in ayahs:
        try:
            wave = fs.to_wave(get(f"https://everyayah.com/data/{EVERYAYAH}/{s:03d}{a:03d}.mp3"))
        except Exception as e:
            log(f"calibration: skip {s}:{a} ({e})")
            continue
        variants, _ = fs.raw_mistakes(wave, [(s, a)])
        found = fs.combine(variants, [(s, a)], votes=1, min_conf=0.0, calibration={})[f"{s}:{a}"]
        for e in found:
            slips.setdefault(f"{s}:{a}", {}).setdefault(f'{e["positionIndex"]}|{e["errorType"]}', []).append(EVERYAYAH)
        done += 1
        print(f"calibration {s}:{a}  slips {len(found)}      ", end="\r", flush=True)
    out = {k: [[int(w.split("|")[0]), w.split("|")[1]] for w in v] for k, v in slips.items()}
    json.dump(out, open(os.path.join(OUT, "calibration.json"), "w"), ensure_ascii=False, indent=1)
    log(f"calibration.json: {sum(len(v) for v in out.values())} habitual slips in {len(out)} of {done} ayahs")
    return {k: {(w, t) for w, t in v} for k, v in out.items()}


# ------------------------------------------------------------------------- 4. the test reciters
def test_reciters(new_calibration):
    import evaluate as ev
    tok = fs.MODEL.multi_level_tokenizer.get_tokenizer()
    meta = {"vocab": tok.get_vocab(), "blank": tok.pad_token_id, "ayahs": {}}
    arrays, raw = {}, []
    for reciter in TEST_RECITERS:
        for s, n in SURAHS.items():
            for a in range(1, n + 1):
                try:
                    wave = fs.to_wave(get(f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3"))
                except Exception as e:
                    log(f"test: skip {reciter} {s}:{a} ({e})")
                    continue
                lp, fps = mt.emissions(fs.MODEL, wave, fs.SR)
                key = f"{reciter}|{s}|{a}"
                arrays[key] = lp.astype(np.float16)
                meta["ayahs"][key] = {"fps": fps, **ayah_info(s, a)}
                raw.append(((s, a), fs.raw_mistakes(wave, [(s, a)])[0], reciter))
                print(f"test {reciter} {s}:{a}      ", end="\r", flush=True)
    np.savez_compressed(os.path.join(OUT, "test_data.npz"), meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    log("test_data.npz saved")
    cfg = dict(votes=fs.VOTES, min_conf=fs.MIN_CONF, calib_mode=fs.CALIB_MODE, skip_kinds=fs.SKIP_KINDS)
    summary = {"settings": {"votes": fs.VOTES, "min_conf": fs.MIN_CONF, "calib_mode": fs.CALIB_MODE},
               "experts_new_calibration_minshawi": ev.score_experts(raw, calibration=new_calibration, **cfg)[0],
               "experts_old_calibration": ev.score_experts(raw, calibration=fs.CALIBRATION, **cfg)[0],
               "experts_no_filtering": ev.score_experts(raw, votes=1, min_conf=0.0, calibration={}, skip_kinds=set())[0]}
    per = {}
    for name, cal in (("new", new_calibration), ("old", fs.CALIBRATION)):
        _, rows = ev.score_experts(raw, calibration=cal, **cfg)
        per[name] = {r: sum(x["reciter"] == r and x["reported"] == 0 for x in rows) for r in TEST_RECITERS}
    summary["passed_clean_per_reciter"] = per
    try:                    # the benchmark needs the `datasets` package; never install it here: it would change the
        import datasets     # noqa: F401  libraries the running server needs (its huggingface-hub) and break its next start
        bn = ev.bench_raw(parity=1)
        summary["benchmark_new_calibration_minshawi"] = ev.score_bench(bn, calibration=new_calibration, **cfg)[0]
        summary["benchmark_old_calibration"] = ev.score_bench(bn, calibration=fs.CALIBRATION, **cfg)[0]
    except Exception as e:
        summary["benchmark"] = f"not run: {e}"
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "results", "summary.json"), "w"), ensure_ascii=False, indent=2)
    log("results/summary.json saved")


def main():
    t0 = time.time()
    log("1/4 his recordings and what the model hears in them")
    his_recordings()
    log("2/4 calibration on him alone")
    cal = calibrate()
    log("3/4 the test reciters and the evaluation")
    test_reciters(cal)
    log("4/4 packing")
    zpath = OUT.rstrip("/") + ".zip"
    if os.path.exists(zpath):
        os.remove(zpath)
    shutil.make_archive(OUT.rstrip("/"), "zip", OUT)
    log(f"done in {(time.time() - t0) / 60:.0f} min: {zpath}")


if __name__ == "__main__":
    main()
