"""
Calibration: find the model's habitual slips.

Expert reciters read every ayah correctly, so anything the model "hears" as a mistake in their recitation is a
slip of the model at that spot (for example it tends to hear a ث as س in one particular word). Those spots are
saved in calibration.json and the server stops reporting that same mistake type on that same word.

Reciters used here are NOT used for tuning or testing (see evaluate.py), so the test numbers stay honest.

Run once:  !python calibrate.py      (about 10 minutes on a T4)
"""
import json
import os
import urllib.request

import faseeh_server as fs

RECITERS = ["Husary_128kbps", "Minshawy_Murattal_128kbps", "Abdul_Basit_Murattal_192kbps",
            "Abdurrahmaan_As-Sudais_192kbps", "Saood_ash-Shuraym_128kbps", "Ghamadi_40kbps"]
# For the website after the evaluation is done: `!python calibrate.py all` also calibrates on the tuning and test
# reciters (more reciters = more habitual slips known). Only do this AFTER `evaluate.py test`, and report the
# numbers from the evaluation, which never saw these reciters during calibration.
import sys
if len(sys.argv) > 1 and sys.argv[1] == "all":
    RECITERS += ["Hudhaify_128kbps", "Muhammad_Ayyoub_128kbps", "Alafasy_128kbps", "MaherAlMuaiqly128kbps"]
AYAHS = [(s, a) for s, n in {78: 40, 80: 42, 103: 3, 108: 3, 112: 4, 113: 5, 114: 6}.items() for a in range(1, n + 1)]
AYAHS.append((5, 109))          # the benchmark ayah, so the benchmark measures what calibration costs


def everyayah(reciter, s, a):
    url = f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3"
    return fs.to_wave(urllib.request.urlopen(url, timeout=60).read())


def main():
    slips = {}
    for reciter in RECITERS:
        done = 0
        for s, a in AYAHS:
            try:
                wave = everyayah(reciter, s, a)
            except Exception as e:
                print("skip", reciter, s, a, e); continue
            variants, _ = fs.raw_mistakes(wave, [(s, a)])
            # the most permissive setting (one vote, any confidence): catch every slip the model can make here
            found = fs.combine(variants, [(s, a)], votes=1, min_conf=0.0, calibration={})[f"{s}:{a}"]
            for e in found:
                slips.setdefault(f"{s}:{a}", {}).setdefault(f'{e["positionIndex"]}|{e["errorType"]}', []).append(reciter)
            done += 1
            print(f"{reciter}  {s}:{a}  slips {len(found)}", end="     \r")
        print(f"\n{reciter}: {done} ayahs")
    out = {k: [[int(w.split("|")[0]), w.split("|")[1]] for w in v] for k, v in slips.items()}
    json.dump(out, open(os.path.join(fs.HERE, "calibration.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(slips, open(os.path.join(fs.HERE, "calibration_detail.json"), "w"), ensure_ascii=False, indent=1)
    print("habitual slips:", sum(len(v) for v in out.values()), "in", len(out), "of", len(AYAHS), "ayahs")


if __name__ == "__main__":
    main()
