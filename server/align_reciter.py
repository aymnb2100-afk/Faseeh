"""
Word timings for the reciter's surah recordings, made by our own model.

Why: the website highlights each word while the reciter reads, and plays single words. That needs the
time of every word in the exact recording that is played. The timings published by the audio site do not
always fit the file it serves today (surah 78 was re-edited: up to 2.5 s off). Here the Quran Muaalem
phoneme model listens to the recording and every word of the surah is aligned to it (CTC forced
alignment), so the timings always belong to the file we play, for any surah.

It checks itself first: surahs whose published timings are known to fit (e.g. 80) are aligned too and
compared word by word; the result is only worth using if those agree closely.

Run in Colab after the server cell (it reuses the loaded model):
    !python align_reciter.py 78          # surahs to re-time
    !python align_reciter.py 78 80 112   # 80 and 112 also serve as the self-check
Writes reciters.json next to it: upload it to data/ in the repository (it replaces the old one).
"""
import json
import sys
import urllib.request

import numpy as np
import torch

import faseeh_server as fs
from quran_transcript import Aya, quran_phonetizer

SITE = "https://faseeh.space"            # where the current data/reciters.json is published
CHECK = {80, 103, 108, 112, 113, 114}    # published timings verified against the audio (pauses at ayah starts)
WINDOW, MARGIN = 20.0, 2.0                # seconds of audio per model pass, and context dropped at each side


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Faseeh"})
    return urllib.request.urlopen(req, timeout=180).read()


# ---------------------------------------------------------------- the model's phoneme probabilities
@torch.no_grad()
def emissions(wave: np.ndarray):
    """log-probabilities of every phoneme for every model frame of the whole recording (frames x vocab), and frames per second."""
    m = fs.MODEL
    sr, n = fs.SR, len(wave)
    hop = WINDOW - 2 * MARGIN
    parts, fps = [], None
    start = 0.0
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


# ---------------------------------------------------------------- CTC forced alignment (Viterbi)
def ctc_align(lp: np.ndarray, targets: list[int], blank: int):
    """For each target token, the first and last frame where it is heard. lp: frames x vocab (log-probabilities)."""
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
    for t in range(1, T):
        c1 = np.concatenate(([NEG], dp[:-1]))
        c2 = np.concatenate(([NEG, NEG], dp[:-2]))
        c2[~skip] = NEG
        stack = np.stack([dp, c1, c2])
        k = stack.argmax(0)
        dp = stack[k, np.arange(S)] + lp[t, z]
        back[t] = k
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    spans = [[None, None] for _ in range(L)]
    for t, s in enumerate(states):
        if s % 2:
            i = s // 2
            if spans[i][0] is None:
                spans[i][0] = t
            spans[i][1] = t + 1
    return spans, float(dp.max()) / T


# ---------------------------------------------------------------- the words and their phonemes
def word_phonemes(text: str, sura: int):
    """The phonemes of each word of an ayah (Uthmani text), as the model writes them."""
    ref = quran_phonetizer(text, fs.MOSHAF, remove_spaces=True, sura_idx=sura)
    word_of_char, w = [], 0
    for c in text:
        word_of_char.append(w)
        if c == " ":
            w += 1
    n_words = w + 1
    owner = [None] * len(ref.phonemes)
    for i, m in enumerate(ref.mappings):
        if m is None or m.deleted:
            continue
        for j in range(m.pos[0], min(m.pos[1], len(owner))):
            owner[j] = word_of_char[i]
    out = [""] * n_words
    cur = 0
    for j, ph in enumerate(ref.phonemes):
        cur = owner[j] if owner[j] is not None else cur   # a phoneme without a letter of its own stays with the word before
        out[cur] += ph
    return out


def align_surah(sura: int, info: dict, vocab: dict, blank: int):
    audio = get(info["audio"])
    wave = fs.to_wave(audio)
    lp, fps = emissions(wave)
    targets, owners = [], []                              # owners: (ayah, occurrence index, word index) per token
    order = {}
    for key in sorted(info["ayahs"], key=int):
        a = int(key)
        words = word_phonemes(Aya(sura_idx=sura, aya_idx=a).get().uthmani, sura)
        old = info["ayahs"][key]
        seq = [n - 1 for n, _, _ in old["seq"]] if old.get("seq") else list(range(len(words)))   # the reciter's repeats
        order[a] = (seq, len(words))
        for occ, w in enumerate(seq):
            for ph in words[w]:
                if ph in vocab:
                    targets.append(vocab[ph]); owners.append((a, occ, w))
    spans, quality = ctc_align(lp, targets, blank)
    ms = lambda f: int(round(f / fps * 1000))
    occ_span = {}
    for (a, occ, w), (f0, f1) in zip(owners, spans):
        if f0 is None:
            continue
        s, e = occ_span.get((a, occ), (f0, f1))
        occ_span[(a, occ)] = (min(s, f0), max(e, f1))
    ayahs = {}
    for a, (seq, n_words) in order.items():
        occs = [(occ, w, occ_span.get((a, occ))) for occ, w in enumerate(seq)]
        steps = []
        for i, (occ, w, sp) in enumerate(occs):
            nxt = next((x[2] for x in occs[i + 1:] if x[2]), None)
            s = ms(sp[0]) if sp else (steps[-1][2] if steps else 0)
            e = ms(nxt[0]) if nxt else (ms(sp[1]) if sp else s)          # contiguous, like the published timings
            steps.append([w + 1, s, max(e, s)])
        words = [None] * n_words
        for n, s, e in steps:
            if words[n - 1] is None:
                words[n - 1] = [s, e]
        for i in range(n_words):                          # a word the model could not place: share its neighbour's time
            if words[i] is None:
                words[i] = list(words[i - 1]) if i else [steps[0][1], steps[0][1]]
        entry = {"from": steps[0][1], "to": steps[-1][2], "words": words}
        if len(seq) != n_words or seq != sorted(seq):
            entry["seq"] = steps
        ayahs[str(a)] = entry
    return ayahs, quality, len(wave) / fs.SR


def compare(new: dict, old: dict):
    diffs = []
    for k, v in old["ayahs"].items():
        for (s0, _), (s1, _) in zip(v["words"], new[k]["words"]):
            diffs.append(abs(s0 - s1))
    d = np.array(diffs)
    return float(np.median(d)), float((d <= 100).mean() * 100), float((d <= 250).mean() * 100)


def main():
    suras = [int(x) for x in sys.argv[1:]] or [78]
    rec = json.loads(get(SITE + "/data/reciters.json"))
    reciter = rec["alafasy"]
    tok = fs.MODEL.multi_level_tokenizer.get_tokenizer()
    vocab, blank = tok.get_vocab(), tok.pad_token_id
    ok_check = True
    for s in suras:
        info = reciter["surahs"][str(s)]
        new, quality, dur = align_surah(s, info, vocab, blank)
        if s in CHECK:
            med, p100, p250 = compare(new, info)
            print(f"self-check {s}: median difference {med:.0f} ms, {p100:.0f}% of words within 0.1 s, {p250:.0f}% within 0.25 s")
            ok_check = ok_check and med <= 80 and p250 >= 90
        else:
            info["ayahs"] = new
            info.pop("exact", None); info.pop("note", None)
            info["timings"] = "aligned by Faseeh (Quran Muaalem, CTC forced alignment) on this exact recording"
            print(f"surah {s}: {sum(len(v['words']) for v in new.values())} words timed over {dur:.0f} s (alignment score {quality:.2f})")
    json.dump(rec, open("reciters.json", "w"), ensure_ascii=False, separators=(",", ":"))
    if ok_check:
        print("\nOK - upload reciters.json to the data folder of the repository.")
    else:
        print("\nThe self-check did not agree closely enough: do NOT upload. Send the lines above.")


if __name__ == "__main__":
    main()
