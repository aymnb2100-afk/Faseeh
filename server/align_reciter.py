"""
Word timings for the reciter's surah recordings, made by our own model and checked against the sound itself.

Why: the website highlights each word while the reciter reads, and plays single words. That needs the time
of every word in the exact recording the website plays. Timings published elsewhere were made for other
edits of the files (surah 78 was off by up to 2.5 s) and start each word about 0.1 s before it is heard.

How, for any surah:
  1. The Quran Muaalem phoneme model listens to the recording (phoneme probabilities every 40 ms).
  2. Every word of the surah is placed on it by CTC forced alignment (Viterbi), phoneme by phoneme.
  3. Phrases the reciter repeats are found by themselves: an ayah that fits badly is tried with every
     possible repeated span and the one the sound supports is kept.
  4. The last word of each ayah lasts until the voice actually stops.
  5. The result is checked against the sound: after every pause, the next word must start where the voice
     starts again, and every ayah must start right after a quiet moment. A surah that does not pass is
     marked "exact": false, so the website highlights whole ayahs there instead of guessing words.

Colab (after the server cell, it reuses the loaded model):
    !python align_reciter.py 78 80 103          # the surahs to time; writes reciters.json
Offline, from a saved model output (dump_align.py) and the mp3 files in ./audio:
    python align_reciter.py --data align_data.npz 78 80
Upload the reciters.json it writes to the data folder of the repository.
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

SITE = "https://faseeh.space"            # where the website and its audio/ and data/ folders are published
WINDOW, MARGIN = 20.0, 2.0                # seconds of audio per model pass, and context dropped at each side
FRAME_MS = 10                             # loudness is measured every 10 ms


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Faseeh"})
    return urllib.request.urlopen(req, timeout=180).read()


# ---------------------------------------------------------------- the model (Colab only)
def emissions(wave: np.ndarray):
    """log-probabilities of every phoneme for every model frame of the whole recording (frames x vocab), and frames per second."""
    import torch
    import faseeh_server as fs
    m = fs.MODEL
    sr, n = fs.SR, len(wave)
    hop = WINDOW - 2 * MARGIN
    parts, fps, start = [], None, 0.0
    with torch.no_grad():
        while True:
            a, b = int(start * sr), min(n, int((start + WINDOW) * sr))
            feats = m.processor([wave[a:b]], sampling_rate=sr, return_tensors="pt")
            feats = {k: v.to(m.device, dtype=m.dtype) for k, v in feats.items()}
            logits = m.model(**feats, return_dict=False)[0]["phonemes"][0].float()
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
            if fps is None:
                fps = lp.shape[0] / ((b - a) / sr)
            first, last = start == 0.0, b >= n
            lo = 0 if first else int(round(MARGIN * fps))
            hi = lp.shape[0] if last else int(round((WINDOW - MARGIN) * fps))
            parts.append(lp[lo:hi])
            if last:
                break
            start += hop
    return np.concatenate(parts), fps


def phonetize(sura: int, aya: int):
    """The Uthmani text of an ayah, its phonemes, and the letters each phoneme comes from."""
    import faseeh_server as fs
    from quran_transcript import Aya, quran_phonetizer
    text = Aya(sura_idx=sura, aya_idx=aya).get().uthmani
    ref = quran_phonetizer(text, fs.MOSHAF, remove_spaces=True, sura_idx=sura)
    maps = []
    for m in ref.mappings:
        pos = getattr(m, "pos", None) if m is not None else None
        maps.append([int(pos[0]), int(pos[1]), bool(getattr(m, "deleted", False))] if pos is not None else None)
    return {"text": text, "phonemes": list(ref.phonemes), "mappings": maps}


# ---------------------------------------------------------------- words and their phonemes
def word_phonemes(ayah: dict):
    """The phonemes of each word of an ayah, as lists of the model's symbols."""
    text, ph, maps = ayah["text"], ayah["phonemes"], ayah["mappings"]
    word_of_char, w = [], 0
    for c in text:
        word_of_char.append(w)
        if c == " ":
            w += 1
    owner = [None] * len(ph)
    for i, m in enumerate(maps):
        if m is None or m[2]:
            continue
        for j in range(m[0], min(m[1], len(owner))):
            owner[j] = word_of_char[i]
    out, cur = [[] for _ in range(w + 1)], 0
    for j, p in enumerate(ph):
        cur = owner[j] if owner[j] is not None else cur   # a phoneme without a letter of its own stays with the word before
        out[cur].append(p)
    return out


# ---------------------------------------------------------------- CTC forced alignment (Viterbi)
def ctc_align(lp: np.ndarray, targets: list, blank: int):
    """The best path through the frames: for each frame, the state (even = blank, odd = token (state-1)/2), and its log-probability."""
    T, L = lp.shape[0], len(targets)
    z = np.full(2 * L + 1, blank, dtype=np.int64)
    z[1::2] = targets
    S = len(z)
    skip = np.zeros(S, dtype=bool)                       # s may come from s-2: a token after a different token
    skip[3::2] = z[3::2] != z[1:-2:2]
    NEG = -1e30
    dp = np.full(S, NEG)
    dp[0], dp[1] = lp[0, z[0]], lp[0, z[1]]
    back = np.zeros((T, S), dtype=np.int8)
    em = lp[:, z]
    for t in range(1, T):
        c1 = np.concatenate(([NEG], dp[:-1]))
        c2 = np.concatenate(([NEG, NEG], dp[:-2]))
        c2[~skip] = NEG
        stack = np.stack([dp, c1, c2])
        k = stack.argmax(0)
        dp = stack[k, np.arange(S)] + em[t]
        back[t] = k
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    return states, em[np.arange(T), states]


class Surah:
    def __init__(self, lp, fps, ayahs, vocab, blank):
        self.lp, self.fps, self.vocab, self.blank = lp, fps, vocab, blank
        self.words = {a: word_phonemes(v) for a, v in ayahs.items()}
        self.seqs = {a: list(range(len(w))) for a, w in self.words.items()}

    def tokens(self, ayahs, seqs=None):
        seqs = seqs or self.seqs
        t, o = [], []
        for a in ayahs:
            for occ, w in enumerate(seqs[a]):
                for p in self.words[a][w]:
                    if p in self.vocab:
                        t.append(self.vocab[p]); o.append((a, occ, w))
        return t, o

    def align(self):
        targets, self.owners = self.tokens(sorted(self.words))
        self.states, self.val = ctc_align(self.lp, targets, self.blank)
        self.first = {}                                  # first frame of each word occurrence (ayah, occ)
        self.last = {}                                   # frame after the last token of each occurrence
        self.ayah_of_frame = np.empty(len(self.states), dtype=np.int64)
        cur = self.owners[0][0]
        for t, s in enumerate(self.states):
            if s % 2:
                a, occ, _ = self.owners[s // 2]
                self.first.setdefault((a, occ), t)
                self.last[(a, occ)] = t + 1
                cur = a
            self.ayah_of_frame[t] = cur

    def ayah_fit(self):
        return {a: float(self.val[self.ayah_of_frame == a].mean()) if (self.ayah_of_frame == a).any() else -9.0 for a in self.words}

    def find_repeats(self):
        """Ayahs that fit badly are tried with every span the reciter might have read twice."""
        fit = self.ayah_fit()
        bar = min(-0.08, 3 * float(np.median(list(fit.values()))))
        changed = []
        order = sorted(self.words)
        for a in order:
            if fit[a] >= bar or len(self.words[a]) < 2:
                continue
            i = order.index(a)
            near = order[max(0, i - 1): i + 2]
            starts = [self.first.get((b, 0)) for b in near]
            f0 = starts[0] if starts[0] is not None else 0
            nxt = self.first.get((order[i + 2], 0)) if i + 2 < len(order) else None
            f1 = nxt if nxt is not None else len(self.lp)
            seg = self.lp[f0:f1]

            def score(seq):
                t, _ = self.tokens(near, {**self.seqs, a: seq})
                return float(ctc_align(seg, t, self.blank)[1].sum()) if len(t) and 2 * len(t) + 1 <= 2 * len(seg) else -1e30
            n = len(self.words[a])
            base = best = score(self.seqs[a])
            best_seq = None
            for x in range(n):
                for y in range(x, n):
                    seq = list(range(y + 1)) + list(range(x, n))
                    v = score(seq)
                    if v > best:
                        best, best_seq = v, seq
            if best_seq and best - base > 30:                # clearly better, not noise
                self.seqs[a] = best_seq
                changed.append((a, best_seq))
        if changed:
            self.align()
        return changed


# ---------------------------------------------------------------- loudness of the recording
def loudness(wave: np.ndarray, sr: int):
    hop = sr * FRAME_MS // 1000
    n = len(wave) // hop
    e = np.sqrt((wave[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(1))
    return e / (np.median(e) or 1)


def pauses(loud, quiet=0.15, min_ms=200):
    q = loud < quiet
    out, i = [], 0
    while i < len(q):
        if q[i]:
            j = i
            while j < len(q) and q[j]:
                j += 1
            if (j - i) * FRAME_MS >= min_ms:
                out.append((i * FRAME_MS, j * FRAME_MS))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------- the timings the website uses
def word_times(sur: Surah, loud):
    ms = lambda f: int(round(f / sur.fps * 1000))
    order = sorted(sur.words)
    starts = {a: ms(sur.first[(a, 0)]) for a in order if (a, 0) in sur.first}
    ayahs = {}
    for i, a in enumerate(order):
        seq = sur.seqs[a]
        occs = [(occ, w, sur.first.get((a, occ))) for occ, w in enumerate(seq)]
        steps = []
        for k, (occ, w, f) in enumerate(occs):
            s = ms(f) if f is not None else (steps[-1][2] if steps else 0)
            nxt = next((x[2] for x in occs[k + 1:] if x[2] is not None), None)
            if nxt is not None:
                e = ms(nxt)                                   # contiguous: a word lasts until the next one starts
            else:                                             # the last word lasts until the voice stops
                e = ms(sur.last.get((a, occ), f or 0))
                limit = next((starts[b] for b in order[i + 1:] if b in starts), len(loud) * FRAME_MS)
                f10 = e // FRAME_MS
                while f10 < len(loud) and f10 * FRAME_MS < limit and loud[f10: f10 + 10].mean() >= 0.15:
                    f10 += 1
                e = min(max(e, f10 * FRAME_MS), limit)
            steps.append([w + 1, s, max(e, s)])
        words = [None] * len(sur.words[a])
        for n, s, e in steps:
            if words[n - 1] is None:
                words[n - 1] = [s, e]
        for k in range(len(words)):                           # a word the model could not place: share its neighbour's time
            if words[k] is None:
                words[k] = list(words[k - 1]) if k else [steps[0][1], steps[0][1]]
        entry = {"from": steps[0][1], "to": steps[-1][2], "words": words}
        if seq != list(range(len(words))):
            entry["seq"] = steps
        ayahs[str(a)] = entry
    return ayahs


def verify(sur: Surah, ayahs: dict, loud):
    """Checks the timings against the sound itself. Returns (passed, numbers)."""
    starts = np.array(sorted(s for v in ayahs.values() for _, s, _ in (v.get("seq") or [[0, w[0], 0] for w in v["words"]])))
    begin = starts[0] - 100
    quiet = max(0.15, 1.2 * float(np.percentile(loud, 3)))   # some recordings keep a little room sound in their pauses
    end = starts[-1] + 100                                 # the silence after the last word has no word after it
    d = np.array([starts[np.argmin(abs(starts - e))] - e for b, e in pauses(loud, quiet, 350) if begin < e <= end and b > 0])   # long enough to be a breath, not a held stop
    quiet_before = []
    for k, v in ayahs.items():
        if int(k) == min(int(x) for x in ayahs):
            continue
        f = v["from"] // FRAME_MS
        quiet_before.append(bool(loud[max(0, f - 40): f + 2].min() < quiet + 0.1))
    fit = float(sur.val.mean())
    stats = {
        "pauses": int(len(d)),
        "median_ms": float(np.median(abs(d))) if len(d) else 0.0,
        "within_100ms": float((abs(d) <= 100).mean()) if len(d) else 1.0,
        "ayah_starts_after_quiet": float(np.mean(quiet_before)) if quiet_before else 1.0,
        "fit": fit,
    }
    ok = stats["median_ms"] <= 80 and stats["within_100ms"] >= 0.8 and stats["ayah_starts_after_quiet"] >= 0.9 and fit >= -0.25
    return ok, stats


def time_surah(lp, fps, ayah_phonemes, wave, sr, vocab, blank):
    sur = Surah(lp, fps, ayah_phonemes, vocab, blank)
    sur.align()
    repeats = sur.find_repeats()
    loud = loudness(wave, sr)
    ayahs = word_times(sur, loud)
    ok, stats = verify(sur, ayahs, loud)
    return ayahs, ok, stats, repeats


def decode(path: str, sr: int = 16000):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


def main():
    args = sys.argv[1:]
    data = None
    if args[:1] == ["--data"]:
        data, args = np.load(args[1]), args[2:]
        meta = json.loads(str(data["meta"]))
    suras = [int(x) for x in args] or [78]
    rec = json.loads(get(SITE + "/data/reciters.json")) if not os.path.exists("reciters.base.json") else json.load(open("reciters.base.json"))
    reciter = rec["alafasy"]
    if data is None:
        import faseeh_server as fs
        tok = fs.MODEL.multi_level_tokenizer.get_tokenizer()
        vocab, blank, sr = tok.get_vocab(), tok.pad_token_id, fs.SR
    else:
        vocab, blank, sr = meta["vocab"], meta["blank"], meta["sr"]
    all_ok = True
    for s in suras:
        info = reciter["surahs"][str(s)]
        if data is None:
            import faseeh_server as fs
            try:
                audio = get(f"{SITE}/audio/{s}.mp3")             # the exact file the website plays
            except Exception:
                audio = get(info["audio"])
            wave = fs.to_wave(audio)
            print(f"surah {s}: {len(wave) / sr:.0f} s of audio, listening ...", flush=True)
            lp, fps = emissions(wave)
            ayah_ph = {int(k): phonetize(s, int(k)) for k in info["ayahs"]}
        else:
            lp, fps = data[f"lp{s}"].astype(np.float64), meta["surahs"][str(s)]["fps"]
            ayah_ph = {int(k): v for k, v in meta["surahs"][str(s)]["ayahs"].items()}
            wave = decode(os.path.join("audio", f"{s}.mp3"), sr)
        ayahs, ok, st, repeats = time_surah(lp, fps, ayah_ph, wave, sr, vocab, blank)
        info["ayahs"] = ayahs
        info.pop("note", None)
        info["timings"] = "aligned by Faseeh (Quran Muaalem, CTC forced alignment) on this exact recording, checked against its pauses"
        if ok:
            info.pop("exact", None)
        else:
            info["exact"] = False
            all_ok = False
        print(f"surah {s}: {'PASSED' if ok else 'NOT exact (whole ayahs will be highlighted)'} - "
              f"{st['pauses']} pauses: words start {st['median_ms']:.0f} ms from where the voice starts (median), "
              f"{st['within_100ms'] * 100:.0f}% within 0.1 s; ayahs after a quiet moment {st['ayah_starts_after_quiet'] * 100:.0f}%; "
              f"fit {st['fit']:.3f}" + (f"; repeats found in ayahs {[a for a, _ in repeats]}" if repeats else ""), flush=True)
    json.dump(rec, open("reciters.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("\nDone - upload reciters.json to the data folder of the repository." if all_ok else
          "\nDone - surahs marked NOT exact stay safe (whole ayahs). Upload reciters.json, and send these lines.")


if __name__ == "__main__":
    main()
