"""
Long madds judged in seconds against the reference reciter (decided with the team's Quran teacher, 2026-11-09).

The munfasil, the muttasil (inside an ayah) and the lazim madd are right when the reader holds them within
TOLERANCE seconds (half a second) of how long the reference reciter (Muhammad Siddiq al-Minshawi, murattal)
holds the very same madd. The natural madd is not timed (it must stay two counts, the model checks it), nor are
the madds at a stop (aared, leen: 2, 4 or 6 counts are all right) and a muttasil on the last word of an ayah
(at a stop it may be 4, 5 or 6).

How a madd is timed, the same way for the reciter and for the reader:
  the recording is placed on its phonemes by CTC forced alignment (Viterbi) on the Quran Muaalem model's
  phoneme probabilities, and the madd lasts from the vowel before it to the sound after it.
madd_reference.json holds the reciter's times, made once from his recordings (build_reference); without it
the server keeps judging long madds by the model's counts, as before.
"""
import json
import os

import numpy as np

MADD = set("اۥۦ")
TOLERANCE = 0.5
WINDOW, MARGIN = 20.0, 2.0            # seconds of audio per model pass, and context dropped at each side


# ------------------------------------------------------------------------------------ the model's phonemes
def emissions(model, wave: np.ndarray, sr: int, lock=None):
    """log-probabilities of every phoneme for every model frame of the recording (frames x vocab), and frames per second."""
    import contextlib
    import torch
    hop = WINDOW - 2 * MARGIN
    parts, fps, start, n = [], None, 0.0, len(wave)
    with torch.no_grad():
        while True:
            a, b = int(start * sr), min(n, int((start + WINDOW) * sr))
            with (lock or contextlib.nullcontext()):
                feats = model.processor([wave[a:b]], sampling_rate=sr, return_tensors="pt")
                feats = {k: v.to(model.device, dtype=model.dtype) for k, v in feats.items()}
                logits = model.model(**feats, return_dict=False)[0]["phonemes"][0].float()
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
            if fps is None:
                fps = lp.shape[0] / max((b - a) / sr, 1e-6)
            first, last = start == 0.0, b >= n
            lo = 0 if first else int(round(MARGIN * fps))
            hi = lp.shape[0] if last else int(round((WINDOW - MARGIN) * fps))
            parts.append(lp[lo:hi])
            if last:
                break
            start += hop
    return np.concatenate(parts), fps


def ctc_align(lp: np.ndarray, targets: list, blank: int):
    """The best path through the frames: for each frame, the state (even = blank, odd = token (state-1)/2), and its log-probability."""
    T, L = lp.shape[0], len(targets)
    z = np.full(2 * L + 1, blank, dtype=np.int64)
    z[1::2] = targets
    S = len(z)
    skip = np.zeros(S, dtype=bool)
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


# ------------------------------------------------------------------------------------ which madds, where
def phoneme_words(text: str, mappings, n_phonemes: int | None = None) -> list:
    """For every phoneme of a text, the word it belongs to (0-based). `mappings` holds, for every character of
    the text, (start, end, deleted) of its phonemes, or objects with .pos and .deleted (quran-transcript).
    A phoneme without a letter of its own stays with the word before it."""
    word_of_char, w = [], 0
    for c in text:
        word_of_char.append(w)
        if c == " ":
            w += 1
    spans = []
    for m in mappings:
        if m is None:
            spans.append(None)
        elif isinstance(m, (list, tuple)):
            spans.append((int(m[0]), int(m[1]), bool(m[2])) if len(m) >= 3 else None)
        else:
            pos = getattr(m, "pos", None)
            spans.append((int(pos[0]), int(pos[1]), bool(getattr(m, "deleted", False))) if pos is not None else None)
    n = n_phonemes if n_phonemes is not None else max((s[1] for s in spans if s), default=0)
    owner = [None] * n
    for i, s in enumerate(spans):
        if s is None or s[2]:
            continue
        for j in range(s[0], min(s[1], n)):
            owner[j] = word_of_char[i]
    out, cur = [], 0
    for o in owner:
        cur = o if o is not None else cur
        out.append(cur)
    return out


def long_madds(phonemes: list, places: list) -> list:
    """The madds timed in seconds. `places[i]` = (ayah key, word, is the last word of its ayah) of phoneme i.
    Returns [(first phoneme of the madd, phoneme after it, (ayah key, word, k))], k counting such madds in a word."""
    found, seen, i, n = [], {}, 0, len(phonemes)
    while i < n:
        c = phonemes[i]
        if c not in MADD:
            i += 1
            continue
        j = i
        while j < n and phonemes[j] == c:
            j += 1
        key, word, last = places[i]
        nxt = phonemes[j] if j < n else None
        same_word = j < n and places[j][:2] == (key, word)
        timed = False
        if j - i >= 6:                                           # lazim (6 counts)
            timed = nxt is not None
        elif j - i >= 4 and nxt == "ء" and i > 0:                # munfasil / muttasil / silah kubra
            timed = not (last and same_word)                     # a muttasil at an ayah's end: free at a stop
        if timed:
            k = seen.get((key, word), 0)
            seen[(key, word)] = k + 1
            found.append((i, j, (key, word, k)))
        i = j
    return found


def durations(lp: np.ndarray, fps: float, phonemes: list, vocab: dict, blank: int, madds: list):
    """Seconds of each madd (from the vowel before it to the sound after it) and how well the recording fits there."""
    idx = [k for k, p in enumerate(phonemes) if p in vocab]
    tok_of = {k: t for t, k in enumerate(idx)}
    if not idx or 2 * len(idx) + 1 > 2 * lp.shape[0]:
        return [None] * len(madds)
    states, val = ctc_align(lp, [vocab[phonemes[k]] for k in idx], blank)
    first = {}
    for t, st in enumerate(states):
        if st % 2:
            first.setdefault(st // 2, t)
    out = []
    for i, j, _ in madds:
        a, b = tok_of.get(i - 1), tok_of.get(j)
        if a is None or b is None or a not in first or b not in first:
            out.append(None)
            continue
        f0, f1 = first[a], first[b]
        fit = float(val[f0:f1 + 1].mean()) if f1 >= f0 else -99.0
        out.append(((f1 - f0) / fps, fit))
    return out


def gap_before(lp: np.ndarray, fps: float, phonemes: list, vocab: dict, blank: int, n: int):
    """Between which two moments (seconds) phoneme number `n` of what was heard begins: the last moment the sound
    before it is recognised and the first moment it is itself. None when the recording cannot be placed."""
    idx = [k for k, p in enumerate(phonemes) if p in vocab]
    t = sum(1 for k in idx if k < n)                             # tokens before the place
    if not 0 < t < len(idx) or 2 * len(idx) + 1 > 2 * lp.shape[0]:
        return None
    states, _ = ctc_align(lp, [vocab[phonemes[k]] for k in idx], blank)
    before, after = np.nonzero(states == 2 * t - 1)[0], np.nonzero(states == 2 * t + 1)[0]
    if not len(before) or not len(after):
        return None
    return (int(before[-1]) + 1) / fps, int(after[0]) / fps


# ------------------------------------------------------------------------------------ the reciter's times
def build_reference(surahs: dict, align, vocab: dict, blank: int) -> dict:
    """surahs: {surah: (lp, fps, {ayah: {"text", "phonemes", "mappings"}})}. `align(lp, fps, ayahs)` returns, for
    every ayah, the frame where each of its phonemes is first heard (None if never), e.g. from align_reciter's
    Surah (which also handles repeated phrases). Returns {"s:a": {"word:k": seconds}}."""
    ref = {}
    for s, (lp, fps, ayahs) in surahs.items():
        first = align(lp, fps, ayahs)
        for a, v in ayahs.items():
            ph = v["phonemes"]
            words = phoneme_words(v["text"], v["mappings"], len(ph))
            nw = len(v["text"].split(" "))
            places = [(f"{s}:{a}", w, w == nw - 1) for w in words]
            for i, j, (key, word, k) in long_madds(ph, places):
                f0, f1 = first[a][i - 1], first[a][j] if j < len(ph) else None
                if f0 is not None and f1 is not None and f1 > f0:
                    ref.setdefault(key, {})[f"{word}:{k}"] = round((f1 - f0) / fps, 3)
    return ref


def load_reference(folder: str) -> dict:
    try:
        return json.load(open(os.path.join(folder, "madd_reference.json"), encoding="utf-8"))
    except FileNotFoundError:
        return {}


# ------------------------------------------------------------------------------------ the reader
def check(lp: np.ndarray, fps: float, vocab: dict, blank: int, reference: dict, tolerance: float,
          run_phonemes: list, run_places: list, min_fit: float = -2.5) -> dict:
    """The reader's long madds against the reciter's. run_phonemes / run_places: the reference phonemes of each
    run of ayahs read without stopping, and the (ayah key, word, last word) of each phoneme.
    Returns {(ayah key, word): [{"k", "phoneme", "seconds", "reference", "ok"}]} for every madd that could be timed."""
    phonemes = [p for run in run_phonemes for p in run]
    places = [p for run in run_places for p in run]
    madds = [m for m in long_madds(phonemes, places) if m[2][0] in reference and f"{m[2][1]}:{m[2][2]}" in reference[m[2][0]]]
    if not madds:
        return {}
    out = {}
    for (i, j, (key, word, k)), d in zip(madds, durations(lp, fps, phonemes, vocab, blank, madds)):
        if d is None or d[1] < min_fit:                          # the recording does not fit here (a word skipped?)
            continue
        want = reference[key][f"{word}:{k}"]
        out.setdefault((key, word), []).append({"k": k, "phoneme": i, "seconds": round(d[0], 2), "reference": want,
                                                "ok": abs(d[0] - want) <= tolerance})
    return out
