"""
Faseeh evaluation: the numbers for the report.

Data is split so the test numbers are honest:
  calibration  Husary, Minshawy, Abdul Basit, Sudais, Shuraym, Ghamdi (calibrate.py) -> habitual slips
  tuning       Hudhaify, Muhammad Ayyoub + even qdat_bench samples -> `sweep` picks the settings
  test         Alafasy, Maher al-Muaiqly + odd qdat_bench samples  -> `test`, the numbers to report

Commands (run in Colab, in this order):
  !python evaluate.py sweep      try every setting on the tuning data, write config.json and results/sweep.csv
  !python evaluate.py test       final numbers on the test data with config.json
  !python evaluate.py own        your own recordings (own_recordings/labels.csv, see below)

own_recordings/labels.csv: one recording per file, of a whole surah (as the website records it), no cutting.
One row per mistake made on purpose; a correct reading gets one row with ayah and word empty.
Words are counted from 1 within the ayah, as in the mushaf.
  file,surah,ayah,word,letter
  ali_112_ok.m4a,112,,,                correct reading of the whole surah
  ali_112_err.m4a,112,1,4,ح            mistake: ayah 1, word 4 (أَحَدٌ), letter ح
  ali_112_err.m4a,112,3,2,             second mistake in the same recording
Optional columns from,to: a recording of only some ayahs (default: the whole surah).

Metrics
  expert pass rate   share of correctly recited ayahs with no mistake reported (false alarms = 1 - this)
  precision          of the mistakes reported, how many were real
  recall             of the real mistakes, how many were reported
  F1                 balance of the two
  PER                phoneme error rate of the model's transcription
"""
import csv
import itertools
import json
import os
import sys
import urllib.request
from collections import Counter

import Levenshtein as lv
import numpy as np

import faseeh_server as fs
from quran_transcript import Aya, explain_error, quran_phonetizer

OUT = os.environ.get("FASEEH_RESULTS", "results")
os.makedirs(OUT, exist_ok=True)
APP_AYAHS = [(s, a) for s, n in {78: 40, 80: 42, 103: 3, 108: 3, 112: 4, 113: 5, 114: 6}.items() for a in range(1, n + 1)]
TUNE_RECITERS = ["Hudhaify_128kbps", "Muhammad_Ayyoub_128kbps"]
TEST_RECITERS = ["Alafasy_128kbps", "MaherAlMuaiqly128kbps"]
BENCH = (5, 109)
GRID = {"votes": [2, 3], "min_conf": [0.0, 0.5, 0.7, 0.8, 0.9, 0.95], "calibration": [False, True],
        "calib_mode": ["exact", "word"], "skip_kinds": [(), ("tashkeel",)]}
TARGET_PASS = float(os.environ.get("FASEEH_TARGET_PASS", "1.0"))
MIN_RECALL = float(os.environ.get("FASEEH_MIN_RECALL", "0.75"))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def write_csv(name, rows):
    if rows:
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def save_summary(key, value):
    path = os.path.join(OUT, "summary.json")
    s = json.load(open(path)) if os.path.exists(path) else {}
    s[key] = value
    json.dump(s, open(path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(value, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- data
def expert_raw(reciters):
    """Model output (all variants, before any filtering) for every app ayah read by each expert reciter."""
    if isinstance(reciters, str):
        reciters = [reciters]
    raw = []
    for reciter, (s, a) in itertools.product(reciters, APP_AYAHS):
        try:
            audio = urllib.request.urlopen(f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3", timeout=60).read()
            wave = fs.to_wave(audio)
        except Exception as e:
            print("skip", s, a, e); continue
        if len(wave) / fs.SR > fs.MAX_SECONDS:
            continue
        raw.append(((s, a), fs.raw_mistakes(wave, [(s, a)])[0], reciter))
        print(f"{reciter} {s}:{a}", end="      \r")
    print()
    return raw


def truth_mistakes(text, ref, said):
    out = set()
    for e in explain_error(uthmani_text=text, ref_ph_text=ref.phonemes, predicted_ph_text=said, mappings=ref.mappings):
        if fs.keep(e):
            out.add((text[:e.uthmani_pos[0]].count(" "), e.error_type))
    return out


def bench_raw(parity):
    """qdat_bench samples with index % 2 == parity: true mistakes, model output and PER counts."""
    from datasets import load_dataset, Audio
    ds = load_dataset("obadx/qdat_bench", split="train").cast_column("audio", Audio(sampling_rate=fs.SR))
    text = Aya(sura_idx=BENCH[0], aya_idx=BENCH[1]).get().uthmani
    ref = quran_phonetizer(text, fs.MOSHAF, remove_spaces=True, sura_idx=BENCH[0])
    raw = []
    for i, item in enumerate(ds):
        if i % 2 != parity:
            continue
        wave = np.asarray(item["audio"]["array"], dtype=np.float32)
        variants, meta = fs.raw_mistakes(wave, [BENCH])
        truth = item["phonetic_transcript"]
        raw.append({"id": item["id"], "gender": item["gender"], "truth": truth_mistakes(text, ref, truth),
                    "variants": variants, "edits": lv.distance(meta["predicted"], truth), "chars": len(truth)})
        print(f"qdat_bench {i + 1}/{len(ds)}", end="      \r")
    print()
    return raw


# ---------------------------------------------------------------- scoring one setting
def score_experts(raw, **cfg):
    rows = []
    for (s, a), variants, reciter in raw:
        found = fs.combine(variants, [(s, a)], **cfg)[f"{s}:{a}"]
        rows.append({"reciter": reciter, "surah": s, "ayah": a, "reported": len(found),
                     "types": "; ".join(f'{e["positionIndex"]}:{e["errorType"]}' for e in found)})
    clean = sum(r["reported"] == 0 for r in rows)
    return {"ayahs": len(rows), "passed_clean": clean, "pass_rate": round(clean / max(len(rows), 1), 4),
            "false_alarms": sum(r["reported"] for r in rows)}, rows


def score_bench(raw, **cfg):
    tot, by_kind, rows = Counter(), {}, []
    key = f"{BENCH[0]}:{BENCH[1]}"
    for r in raw:
        found = fs.combine(r["variants"], [BENCH], **cfg)[key]
        det = {(e["positionIndex"], e["kind"]) for e in found if e["kind"] != "sifat"}   # sifat are not annotated
        t = r["truth"]
        tot.update(tp=len(t & det), fp=len(det - t), fn=len(t - det))
        for kind in ("normal", "tashkeel", "tajweed"):
            tk, dk = {x for x in t if x[1] == kind}, {x for x in det if x[1] == kind}
            by_kind.setdefault(kind, Counter()).update(tp=len(tk & dk), fp=len(dk - tk), fn=len(tk - dk))
        rows.append({"id": r["id"], "gender": r["gender"], "true": len(t), "reported": len(det),
                     "tp": len(t & det), "fp": len(det - t), "fn": len(t - det)})
    p, rc, f = prf(tot["tp"], tot["fp"], tot["fn"])
    per = round(sum(r["edits"] for r in raw) / max(sum(r["chars"] for r in raw), 1), 4)
    kinds = {k: dict(zip(("precision", "recall", "f1"), prf(c["tp"], c["fp"], c["fn"])), true=c["tp"] + c["fn"])
             for k, c in by_kind.items()}
    return {"samples": len(raw), "PER": per, "precision": p, "recall": rc, "f1": f, "by_type": kinds}, rows


# ---------------------------------------------------------------- commands
def sweep():
    ex, bn = expert_raw(TUNE_RECITERS), bench_raw(parity=0)
    rows = []
    for votes, min_conf, cal, mode, skip in itertools.product(*GRID.values()):
        if not cal and mode == "word":
            continue
        cfg = dict(votes=votes, min_conf=min_conf, calibration=fs.CALIBRATION if cal else {}, calib_mode=mode,
                   skip_kinds=set(skip))
        e, _ = score_experts(ex, **cfg)
        b, _ = score_bench(bn, **cfg)
        rows.append({"votes": votes, "min_conf": min_conf, "calibration": cal, "calib_mode": mode,
                     "skip_kinds": "+".join(skip), "expert_pass": e["pass_rate"], "false_alarms": e["false_alarms"],
                     "precision": b["precision"], "recall": b["recall"], "f1": b["f1"]})
    write_csv("sweep.csv", rows)
    # Fewest false alarms first; among equals the best F1. A setting must still catch most real mistakes.
    usable = [r for r in rows if r["recall"] >= MIN_RECALL] or rows
    ok = [r for r in usable if r["expert_pass"] >= TARGET_PASS]
    best = max(usable, key=lambda r: (r["expert_pass"], r["f1"]))
    cfg = {"votes": best["votes"], "min_conf": best["min_conf"], "calibration": best["calibration"],
           "calib_mode": best["calib_mode"], "skip_kinds": [k for k in best["skip_kinds"].split("+") if k]}
    json.dump(cfg, open(os.path.join(fs.HERE, "config.json"), "w"))
    print("\nsetting  pass  F1")
    for r in sorted(rows, key=lambda r: (-r["expert_pass"], -r["f1"]))[:12]:
        print(f'votes {r["votes"]}  conf {r["min_conf"]:.2f}  calib {str(r["calibration"]):5} {r["calib_mode"]:5}  skip {r["skip_kinds"] or "-":8}'
              f'   pass {r["expert_pass"]:.1%}  false alarms {r["false_alarms"]:3d}  F1 {r["f1"]:.3f}  recall {r["recall"]:.3f}')
    save_summary("sweep_choice", dict(cfg, tuning_expert_pass=best["expert_pass"], tuning_f1=best["f1"],
                                      tuning_recall=best["recall"], target_pass=TARGET_PASS,
                                      met_target=bool(ok), min_recall=MIN_RECALL))


def test():
    cfg = dict(votes=fs.VOTES, min_conf=fs.MIN_CONF, calibration=fs.CALIBRATION if fs.USE_CALIBRATION else {},
               calib_mode=fs.CALIB_MODE, skip_kinds=fs.SKIP_KINDS)
    ex, bn = expert_raw(TEST_RECITERS), bench_raw(parity=1)
    base = dict(votes=1, min_conf=0.0, calibration={}, skip_kinds=set())   # no false-alarm filtering, for comparison
    e0, _ = score_experts(ex, **base); b0, _ = score_bench(bn, **base)
    e1, erows = score_experts(ex, **cfg); b1, brows = score_bench(bn, **cfg)
    write_csv("test_experts.csv", erows); write_csv("test_benchmark.csv", brows)
    per_reciter = {r: dict(zip(("ayahs", "passed_clean"), (sum(x["reciter"] == r for x in erows),
                                                          sum(x["reciter"] == r and x["reported"] == 0 for x in erows))))
                   for r in TEST_RECITERS}
    save_summary("test", {"settings": {"votes": fs.VOTES, "min_conf": fs.MIN_CONF, "calibration": fs.USE_CALIBRATION,
                                       "calib_mode": fs.CALIB_MODE, "skip_kinds": sorted(fs.SKIP_KINDS)},
                          "per_reciter": per_reciter,
                          "without_filtering": {"experts": e0, "benchmark": b0},
                          "final": {"experts": e1, "benchmark": b1}})


def own(folder="own_recordings"):
    """Your own recordings with mistakes made on purpose: do the reported words match the real ones?"""
    labels = os.path.join(folder, "labels.csv")
    if not os.path.exists(labels):
        return save_summary("own", {"note": f"no {labels} yet"})
    counts = {s: n for s, n in {78: 40, 80: 42, 103: 3, 108: 3, 112: 4, 113: 5, 114: 6}.items()}
    files = {}
    for r in csv.DictReader(open(labels, encoding="utf-8-sig")):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("file"):
            continue
        s = int(r["surah"])
        f = files.setdefault(r["file"], {"surah": s, "from": int(r.get("from") or 1),
                                          "to": int(r.get("to") or counts.get(s, 1)), "mistakes": set()})
        if r.get("ayah") and r.get("word"):
            f["mistakes"].add((int(r["ayah"]), int(r["word"])))
    rows, tot = [], Counter()
    for file, f in files.items():
        s, ayahs = f["surah"], [(f["surah"], a) for a in range(f["from"], f["to"] + 1)]
        print(f"{file}: surah {s}, ayahs {f['from']}-{f['to']} ...", flush=True)
        res = fs.analyze_wave(fs.to_wave(open(os.path.join(folder, file), "rb").read()), ayahs)
        found, details = set(), []
        for _, a in ayahs:
            for e in res.get(f"{s}:{a}", []):
                found.add((a, e["positionIndex"] + 1))
                details.append(f'{a}:{e["positionIndex"] + 1} {e["letter"]} {e["errorType"]}')
        real = f["mistakes"]
        tot.update(tp=len(real & found), fp=len(found - real), fn=len(real - found))
        rows.append({"file": file, "surah": s, "mismatch": bool(res.get("_mismatch")),
                     "expected": " ".join(f"{a}:{w}" for a, w in sorted(real)),
                     "found": " ".join(f"{a}:{w}" for a, w in sorted(found)),
                     "caught": len(real & found), "missed": len(real - found), "false_alarms": len(found - real),
                     "details": "; ".join(details)})
        print(f"   real {rows[-1]['expected'] or '-'} | found {rows[-1]['found'] or '-'}"
              + ("   (the model says this is not the selected surah)" if rows[-1]["mismatch"] else ""), flush=True)
    write_csv("own.csv", rows)
    p, r, f1 = prf(tot["tp"], tot["fp"], tot["fn"])
    clean = [x for x in rows if not x["expected"]]
    save_summary("own", {"recordings": len(rows), "mistakes_made": tot["tp"] + tot["fn"], "caught": tot["tp"],
                         "missed": tot["fn"], "false_alarms": tot["fp"], "precision": p, "recall": r, "f1": f1,
                         "correct_readings": len(clean), "correct_passed_clean": sum(not x["found"] for x in clean)})


if __name__ == "__main__":
    {"sweep": sweep, "test": test, "own": own}[sys.argv[1] if len(sys.argv) > 1 else "test"]()
