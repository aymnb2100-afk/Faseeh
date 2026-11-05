"""
Faseeh analysis server: the real AIEngine behind the web page.

POST /analyze  (body = the recorded audio, header X-Ayahs = "112:1,112:2")
  1. converts the audio to 16 kHz mono,
  2. runs the Quran Muaalem model (obadx/muaalem-model-v3_2) to get the phonemes the reader actually said,
  3. builds the correct Hafs phonemes for the selected ayahs with quran-transcript,
  4. aligns the two and returns every mistake with its ayah, word, letter and type.

Response (same shape the page already understands):
  { "112:1": [ {"positionIndex": 3, "letter": "ح", "errorType": "إبدال الحاء هاءً",
               "kind": "normal", "expected": "حَ", "heard": "هَ", "tajweed": {"ar": "...", "en": "..."}} ],
    "_meta": { "predicted": "...", "reference": "...", "seconds": 4.2 } }

Model and phonetic script: Quran Muaalem (https://github.com/obadx/quran-muaalem), QML 1.0 license.
The page must show the licence's AI advisory notice to every user (it does).
"""
import json
import os
import subprocess
import threading
from types import SimpleNamespace
import tempfile
import time

import Levenshtein as lv
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from quran_transcript import Aya, MoshafAttributes, explain_error, quran_phonetizer
from quran_transcript.phonetics.sifa import chunck_phonemes
from quran_muaalem import Muaalem

SR = 16000
MAX_SECONDS = float(os.environ.get("FASEEH_MAX_SECONDS", "330"))   # a whole surah (An-Naba is about 4.5 minutes)
MAX_BYTES = 25 * 1024 * 1024
CHUNK_SECONDS = 20          # long recordings are analysed in pieces cut at pauses
MISMATCH = 0.45             # above this share of differences the reader did not recite the selected ayahs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hafs (via ash-Shatibiyyah) as recited by Mishary Alafasy, the reciter the page plays
MOSHAF = MoshafAttributes(
    rewaya="hafs",
    recitation_speed="murattal",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)

print(f"Loading Quran Muaalem on {DEVICE} ...")
MODEL = Muaalem(device=DEVICE, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
print("Model ready.")

app = FastAPI(title="Faseeh analysis server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ARABIC_LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهوي" "أإآؤئىةٱ")

# Letter swaps the page has a dedicated explanation for (expected letter, heard letter) -> rules.json key
SWAPS = {
    ("ح", "ه"): "إبدال الحاء هاءً",
    ("ص", "س"): "إبدال الصاد سينًا",
    ("ث", "س"): "إبدال الثاء سينًا",
    ("ع", "ء"): "إبدال العين همزة",
    ("ط", "ت"): "ترقيق الطاء",
    ("ق", "ك"): "ترقيق القاف",
}
# Heavy letters: heard light (sifat level) -> rules.json key
LIGHT = {"ق": "ترقيق القاف", "ط": "ترقيق الطاء", "خ": "ترقيق الخاء"}


def to_wave(raw: bytes) -> np.ndarray:
    """Any browser recording (webm/ogg/mp4/wav) -> 16 kHz mono float32, using ffmpeg."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in"), os.path.join(d, "out.wav")
        with open(src, "wb") as f:
            f.write(raw)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SR), dst],
                           capture_output=True)
        if r.returncode != 0:
            raise HTTPException(400, "Could not read the audio: " + r.stderr.decode(errors="ignore")[:200])
        wave, _ = sf.read(dst, dtype="float32")
    return wave


# Half a second of silence on each side: speech models tend to drop the very first and last sound of a clip
# when it starts or ends abruptly (e.g. the first hamzah of an ayah).
PAD = np.zeros(int(0.5 * SR), dtype=np.float32)


def padded(wave: np.ndarray) -> np.ndarray:
    return np.concatenate([PAD, wave.astype(np.float32), PAD])


def keep(err) -> bool:
    """False for differences that are not mistakes in Hafs:
    - the madd at a stop (aared) and the leen madd may be 2, 4 or 6 counts;
    - the separated / connected madd may be 4 or 5 counts, and a one-count difference on a long madd is within
      how reciters and the model measure length.
    The natural madd (2 counts) must stay exact."""
    if err.error_type != "tajweed" or err.expected_len is None or err.predicted_len is None:
        return True
    names = " ".join(r.name.en for r in (err.ref_tajweed_rules or []))
    if "Aared" in names or "Leen" in names:
        return err.predicted_len < 2
    if "Monfasel" in names or "Mottasel" in names or "Lazem" in names:
        return abs(err.expected_len - err.predicted_len) > 1
    return True


def letter_at(text: str, start: int, end: int) -> str:
    """First real letter inside text[start:end], else the nearest letter before it."""
    for i in range(start, max(end, start + 1)):
        if i < len(text) and text[i] in ARABIC_LETTERS:
            return text[i]
    for i in range(min(start, len(text) - 1), -1, -1):
        if text[i] in ARABIC_LETTERS:
            return text[i]
    return ""


def uthmani_index_of_phoneme(mappings, ph_idx: int) -> int:
    for i, m in enumerate(mappings):
        if m is not None and not m.deleted and m.pos[0] <= ph_idx < m.pos[1]:
            return i
    return 0


def error_key(err) -> str:
    """Our explanation key (rules.json) for one ReciterError."""
    if err.error_type == "tajweed":
        names = " ".join(type(r).__name__ for r in (err.ref_tajweed_rules or []) + (err.missing_tajweed_rules or []))
        if "Madd" in names:
            return "خطأ في المد"
        if "Ghonna" in names or "Idgham" in names:
            return "ترك الغنة"
        if "Qalqala" in names:
            return "ترك القلقلة"
        return "خطأ في التجويد"
    if err.error_type == "tashkeel":
        exp, got = err.expected_ph or "", err.preditected_ph or ""
        if exp[:1] in ("ن", "م") and exp[:1] == got[:1] and len(exp) != len(got):
            return "ترك الغنة"                    # the nasal letter was held too short: it is the ghunnah
        return "خطأ في الحركة"
    if err.speech_error_type == "replace":
        exp, got = err.expected_ph[:1], (err.preditected_ph or "")[:1]
        return SWAPS.get((exp, got), "إبدال حرف")
    if err.speech_error_type == "delete":
        return "حذف حرف"
    return "زيادة حرف"


def tajweed_names(err):
    rules = (err.ref_tajweed_rules or []) + (err.missing_tajweed_rules or [])
    return {"ar": "، ".join(r.name.ar for r in rules), "en": ", ".join(r.name.en for r in rules)} if rules else None


def ref_to_pred(ref: str, pred: str) -> list[int]:
    """For every position in the reference phoneme string, the matching position in the predicted one."""
    m = [0] * (len(ref) + 1)
    for tag, i1, i2, j1, j2 in lv.opcodes(ref, pred):
        for k in range(i1, i2 + 1):
            m[k] = j1 if tag == "delete" else min(j2, j1 + (k - i1))
    m[len(ref)] = len(pred)
    return m


def split_predicted(ref_parts: list[str], predicted: str) -> tuple[list[str], list[int]]:
    """Cut the predicted phoneme string at the same places as the per-ayah references.
    Returns the pieces and where each piece starts in the predicted string."""
    m = ref_to_pred("".join(ref_parts), predicted)
    pieces, starts, prev, total = [], [], 0, 0
    for p in ref_parts[:-1]:
        total += len(p)
        at = m[total]
        pieces.append(predicted[prev:at]); starts.append(prev)
        prev = at
    pieces.append(predicted[prev:]); starts.append(prev)
    return pieces, starts


def sifat_errors(ref_out, pred_sifat, pred_text):
    """Heavy letters read light and missing qalqalah, from the model's sifat predictions (equal phoneme groups only)."""
    ref_groups = chunck_phonemes(ref_out.phonemes)
    pred_groups = chunck_phonemes(pred_text)
    if len(pred_groups) != len(pred_sifat) or len(ref_groups) != len(ref_out.sifat):
        return []
    found, ref_pos = [], [0]
    for g in ref_groups:
        ref_pos.append(ref_pos[-1] + len(g))
    for tag, i1, i2, j1, j2 in lv.opcodes([g[0] for g in ref_groups], [g[0] for g in pred_groups]):
        if tag != "equal":
            continue
        for a, b in zip(range(i1, i2), range(j1, j2)):
            ref_s, got = ref_out.sifat[a], pred_sifat[b]
            letter = ref_groups[a][0]
            got_tafkheem = getattr(getattr(got, "tafkheem_or_taqeeq", None), "text", None)
            got_qalqla = getattr(getattr(got, "qalqla", None), "text", None)
            if letter in LIGHT and ref_s.tafkheem_or_taqeeq == "mofakham" and got_tafkheem == "moraqaq":
                found.append((ref_pos[a], letter, LIGHT[letter]))
            elif ref_s.qalqla == "moqalqal" and got_qalqla == "not_moqalqal":
                found.append((ref_pos[a], letter, "ترك القلقلة"))
    return found


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


@app.post("/analyze")
async def analyze(request: Request):
    keys = [k for k in request.headers.get("x-ayahs", "").split(",") if ":" in k]
    if not keys or len(keys) > 60:
        raise HTTPException(400, "Header X-Ayahs is required, e.g. 112:1,112:2")
    try:
        ayahs = [(int(s), int(a)) for s, a in (k.split(":") for k in keys)]
        for s, a in ayahs:
            Aya(sura_idx=s, aya_idx=a)
    except Exception:
        raise HTTPException(400, "Unknown ayah in X-Ayahs.")
    body = await request.body()
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "The recording is too large.")
    wave = await run_in_threadpool(to_wave, body)
    seconds = len(wave) / SR
    if seconds < 0.5:
        raise HTTPException(400, "The recording is empty or too short.")
    if seconds > MAX_SECONDS:
        raise HTTPException(413, f"Please record at most {MAX_SECONDS:.0f} seconds at a time.")

    if float(np.sqrt(np.mean(wave ** 2))) < 1e-3:
        raise HTTPException(422, "silence")             # nothing was recorded (microphone muted?)
    return await run_in_threadpool(analyze_wave, wave, ayahs)


# ---------------------------------------------------------------------------------------------------
# Running the model: one request at a time on the GPU, long recordings in pieces cut at the quietest point.
MODEL_LOCK = threading.Lock()


def pieces(wave: np.ndarray) -> list[np.ndarray]:
    longest = int(CHUNK_SECONDS * SR)
    if len(wave) <= longest:
        return [wave]
    frame = int(0.03 * SR)
    energy = np.convolve(wave.astype(np.float64) ** 2, np.ones(frame) / frame, mode="same")
    out, start = [], 0
    while len(wave) - start > longest:
        lo, hi = start + longest // 2, start + longest
        cut = lo + int(np.argmin(energy[lo:hi]))          # a pause between words or ayahs
        out.append(wave[start:cut]); start = cut
    out.append(wave[start:])
    return out


def run_model(wave: np.ndarray, whole) -> SimpleNamespace:
    """Model output for the whole recording: phoneme text, per-phoneme confidence and sifat, pieces joined."""
    texts, probs, sifat = [], [], []
    for piece in pieces(wave):
        with MODEL_LOCK:
            out = MODEL([padded(piece)], [whole], sampling_rate=SR)[0]
        texts.append(out.phonemes.text)
        probs.extend(float(p) for p in out.phonemes.probs)
        sifat.extend(out.sifat)
    return SimpleNamespace(phonemes=SimpleNamespace(text="".join(texts), probs=probs), sifat=sifat)


# ---------------------------------------------------------------------------------------------------
# Avoiding false alarms. A correct reading must not be marked wrong, so a mistake is only reported when
#  1. the model hears it in at least VOTES of the 3 versions of the recording (normal speed, 5% slower,
#     5% faster): a real mistake is heard in all of them, a slip of the model usually in one only, and
#  2. for a replaced or added sound, the model is at least MIN_CONF sure of the sound it heard.
# The values are chosen with evaluate.py (sweep), which measures both false alarms and missed mistakes.
#  3. the same mistake is not also "heard" when expert reciters read that ayah correctly (calibration.json,
#     made by calibrate.py from reciters that are NOT used for testing): those spots are the model's own
#     habitual slips, not the reader's.
# config.json (written by `evaluate.py sweep` on separate tuning data) overrides the defaults below.
VOTES = int(os.environ.get("FASEEH_VOTES", "2"))
MIN_CONF = float(os.environ.get("FASEEH_MIN_CONF", "0.5"))
USE_CALIBRATION = True
CALIB_MODE = "exact"          # "exact": same word and same mistake type; "word": any mistake on that word
SKIP_KINDS = set()            # mistake kinds not reported at all (e.g. {"tashkeel"} if they are not reliable)
SPEEDS = (1.0, 0.95, 1.05)
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    _cfg = json.load(open(os.path.join(HERE, "config.json")))
    VOTES, MIN_CONF, USE_CALIBRATION = int(_cfg["votes"]), float(_cfg["min_conf"]), bool(_cfg["calibration"])
    CALIB_MODE, SKIP_KINDS = _cfg.get("calib_mode", "exact"), set(_cfg.get("skip_kinds", []))
    print("config.json:", _cfg)
except FileNotFoundError:
    pass
try:
    CALIBRATION = {k: {(w, t) for w, t in v} for k, v in json.load(open(os.path.join(HERE, "calibration.json"))).items()}
    print("calibration.json:", sum(len(v) for v in CALIBRATION.values()), "habitual slips in", len(CALIBRATION), "ayahs")
except FileNotFoundError:
    CALIBRATION = {}


def stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    """The recording played `factor` times as fast (simple resampling)."""
    if factor == 1.0:
        return wave
    n = max(1, int(len(wave) / factor))
    return np.interp(np.linspace(0, len(wave) - 1, n), np.arange(len(wave)), wave).astype(np.float32)


def raw_mistakes(wave: np.ndarray, ayahs: list[tuple[int, int]]):
    """Every difference the model hears, separately for each version of the recording, with its confidence."""
    texts = [Aya(sura_idx=s, aya_idx=a).get().uthmani for s, a in ayahs]
    refs = [quran_phonetizer(t, MOSHAF, remove_spaces=True, sura_idx=s) for t, (s, _) in zip(texts, ayahs)]
    whole = quran_phonetizer(" ".join(texts), MOSHAF, remove_spaces=True, sura_idx=ayahs[0][0])
    t0 = time.time()
    outs = [run_model(stretch(wave, f), whole) for f in SPEEDS]
    infer_s = time.time() - t0

    variants = []
    for v, out in enumerate(outs):
        predicted = out.phonemes.text
        probs = [float(p) for p in out.phonemes.probs]
        if len(probs) != len(predicted):
            probs = None                                   # tokens are not single characters: no confidence
        parts, starts = split_predicted([r.phonemes for r in refs], predicted)
        per_ayah = {}
        for (s, a), text, ref, heard, off in zip(ayahs, texts, refs, parts, starts):
            items = {}
            m = ref_to_pred(ref.phonemes, heard)
            for err in explain_error(uthmani_text=text, ref_ph_text=ref.phonemes, predicted_ph_text=heard, mappings=ref.mappings):
                if not keep(err):
                    continue
                start, end = err.uthmani_pos
                word, letter, key = text[:start].count(" "), letter_at(text, start, end), error_key(err)
                j0, j1 = m[err.ph_pos[0]], m[err.ph_pos[1]]
                conf = min(probs[off + j0: off + j1]) if probs and j1 > j0 else None
                if (word, letter, key) not in items:
                    items[(word, letter, key)] = {"positionIndex": word, "letter": letter, "errorType": key,
                                                  "kind": err.error_type, "speech": err.speech_error_type,
                                                  "expected": err.expected_ph, "heard": err.preditected_ph,
                                                  "tajweed": tajweed_names(err), "conf": conf}
            if len(ayahs) == 1:                            # sifat are only aligned reliably for one ayah at a time
                for ph_idx, letter, key in sifat_errors(ref, out.sifat, heard):
                    ui = uthmani_index_of_phoneme(ref.mappings, ph_idx)
                    word = text[:ui].count(" ")
                    items.setdefault((word, letter, key), {"positionIndex": word, "letter": letter, "errorType": key,
                                                           "kind": "sifat", "speech": "replace", "expected": letter,
                                                           "heard": letter, "tajweed": None, "conf": None})
            per_ayah[(s, a)] = items
        variants.append(per_ayah)
    meta = {"predicted": outs[0].phonemes.text, "reference": whole.phonemes, "inferenceSeconds": round(infer_s, 2),
            "difference": round(lv.distance(whole.phonemes, outs[0].phonemes.text) / max(len(whole.phonemes), 1), 3)}
    return variants, meta


def combine(variants, ayahs, votes=None, min_conf=None, calibration=None, calib_mode=None, skip_kinds=None) -> dict:
    """Keep the mistakes heard in enough versions and, for replaced or added sounds, heard with enough confidence,
    and drop the model's habitual slips on that ayah (calibration)."""
    votes = VOTES if votes is None else votes
    min_conf = MIN_CONF if min_conf is None else min_conf
    calibration = (CALIBRATION if USE_CALIBRATION else {}) if calibration is None else calibration
    calib_mode = CALIB_MODE if calib_mode is None else calib_mode
    skip_kinds = SKIP_KINDS if skip_kinds is None else skip_kinds
    result = {}
    for s, a in ayahs:
        found = {}
        for v in variants:
            for k, item in v[(s, a)].items():
                found.setdefault(k, []).append(item)
        kept, habitual = [], calibration.get(f"{s}:{a}", set())
        habitual_words = {w for w, _ in habitual}
        for k, seen in found.items():
            first = seen[0]
            if len(seen) < votes or first["kind"] in skip_kinds:
                continue
            if (first["positionIndex"], first["errorType"]) in habitual or (
                    calib_mode == "word" and first["positionIndex"] in habitual_words):
                continue
            confs = [x["conf"] for x in seen if x["conf"] is not None]
            if seen[0]["speech"] in ("replace", "insert") and confs and max(confs) < min_conf:
                continue
            item = dict(seen[0]); item["votes"] = len(seen); item["conf"] = max(confs) if confs else None
            kept.append(item)
        result[f"{s}:{a}"] = sorted(kept, key=lambda x: x["positionIndex"])
    return result


def analyze_wave(wave: np.ndarray, ayahs: list[tuple[int, int]]) -> dict:
    """The whole analysis for one recording of the given ayahs (also used by evaluate.py)."""
    variants, meta = raw_mistakes(wave, ayahs)
    if meta["difference"] > MISMATCH:                      # another passage, or only part of it, was recited
        result = {f"{s}:{a}": [] for s, a in ayahs}
        result["_mismatch"] = True
        result["_meta"] = dict(meta, seconds=round(len(wave) / SR, 2), device=DEVICE)
        return result
    result = combine(variants, ayahs)
    result["_meta"] = dict(meta, seconds=round(len(wave) / SR, 2), device=DEVICE, votes=VOTES, minConf=MIN_CONF,
                           calibration=USE_CALIBRATION and bool(CALIBRATION), calibMode=CALIB_MODE,
                           skipKinds=sorted(SKIP_KINDS))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
