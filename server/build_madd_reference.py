"""
Makes madd_reference.json: how long the reference reciter holds every long madd (munfasil, muttasil, lazim)
in the website's surahs, from what the model heard in his recordings (align_data.npz, made by
prepare_reciter.py). The server judges a reader's long madds against these times (madd_timing.py).

    python build_madd_reference.py align_data.npz      -> madd_reference.json (copy it to hf_space/)
"""
import json
import sys

import numpy as np

import align_reciter as ar
import madd_timing as mt


def first_frames(vocab, blank):
    """align(lp, fps, ayahs) -> {ayah: [first frame of each phoneme or None]}, with repeated phrases handled."""
    def align(lp, fps, ayahs):
        sur = ar.Surah(lp, fps, ayahs, vocab, blank)
        sur.align()
        sur.find_repeats()
        _, owners = sur.tokens(sorted(sur.words))
        first_tok = {}
        for t, st in enumerate(sur.states):
            if st % 2:
                first_tok.setdefault(st // 2, t)
        out = {a: [None] * len(v["phonemes"]) for a, v in ayahs.items()}
        offset = {a: np.cumsum([0] + [len(w) for w in sur.words[a]]).tolist() for a in ayahs}
        pos_in_word, prev = 0, None
        for t, (a, occ, w) in enumerate(owners):
            pos_in_word = pos_in_word + 1 if prev == (a, occ) else 0
            prev = (a, occ)
            if sur.seqs[a].index(w) != occ:                  # a repeated phrase: its first reading counts
                continue
            # owners only lists phonemes in the vocabulary; all of the model's phonemes are
            i = offset[a][w] + pos_in_word
            if i < len(out[a]) and t in first_tok:
                out[a][i] = first_tok[t]
        return out
    return align


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "align_data.npz"
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    vocab, blank = meta["vocab"], meta["blank"]
    surahs = {}
    for s, info in meta["surahs"].items():
        ayahs = {int(k): v for k, v in info["ayahs"].items()}
        surahs[int(s)] = (data[f"lp{s}"].astype(np.float64), info["fps"], ayahs)
    ref = mt.build_reference(surahs, first_frames(vocab, blank), vocab, blank)
    json.dump(ref, open("madd_reference.json", "w"), ensure_ascii=False, indent=1)
    secs = [v for a in ref.values() for v in a.values()]
    print(f"madd_reference.json: {len(secs)} long madds in {len(ref)} ayahs, "
          f"median {np.median(secs):.2f} s (from {min(secs):.2f} to {max(secs):.2f} s)")


if __name__ == "__main__":
    main()
