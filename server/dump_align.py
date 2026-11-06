"""
Saves what the Quran Muaalem model hears in the reciter's recordings, so the word timings can be worked
out and checked outside Colab (the model only runs here). One file, align_data.npz:
  - for every surah: the phoneme log-probabilities of every model frame (float16) and frames per second
  - for every ayah: the Uthmani text, its phonemes and which letters each phoneme comes from
  - the model's phoneme vocabulary
Run in Colab after align_reciter.py exists:  !python dump_align.py
"""
import json
import sys

import numpy as np

import align_reciter as ar
import faseeh_server as fs
from quran_transcript import Aya, quran_phonetizer

SURAHS = [int(x) for x in sys.argv[1:]] or [78, 80, 103, 108, 112, 113, 114]


def mapping(m):
    if m is None:
        return None
    pos = getattr(m, "pos", None)
    return [int(pos[0]), int(pos[1]), bool(getattr(m, "deleted", False))] if pos is not None else None


def main():
    tok = fs.MODEL.multi_level_tokenizer.get_tokenizer()
    meta = {"vocab": tok.get_vocab(), "blank": tok.pad_token_id, "sr": fs.SR, "surahs": {}}
    arrays = {}
    for s in SURAHS:
        print(f"surah {s}: downloading ...", flush=True)
        wave = fs.to_wave(ar.get(f"{ar.SITE}/audio/{s}.mp3"))       # the exact file the website plays
        print(f"surah {s}: {len(wave) / fs.SR:.0f} s of audio, listening ...", flush=True)
        lp, fps = ar.emissions(wave)
        arrays[f"lp{s}"] = lp.astype(np.float16)
        ayahs = {}
        for a in range(1, 300):
            try:
                text = Aya(sura_idx=s, aya_idx=a).get().uthmani
            except Exception:
                break
            ref = quran_phonetizer(text, fs.MOSHAF, remove_spaces=True, sura_idx=s)
            ayahs[a] = {"text": text, "phonemes": list(ref.phonemes),
                        "mappings": [mapping(m) for m in ref.mappings]}
        meta["surahs"][s] = {"fps": fps, "seconds": len(wave) / fs.SR, "ayahs": ayahs}
        print(f"surah {s}: done ({lp.shape[0]} frames, {len(ayahs)} ayahs)", flush=True)
    np.savez_compressed("align_data.npz", meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    print("\nSaved align_data.npz - send this file.")


if __name__ == "__main__":
    main()
