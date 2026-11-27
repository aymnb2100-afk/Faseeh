# Faseeh · Handoff

For whoever continues this project after us.

The [README](README.md) says what Faseeh is, how to run it and how it is laid out. This file is the
other half: how the parts fit together at runtime, which rules must not be broken, which knobs change
the results, what is actually proven and what is not, and what we would do next.

---

## 1. The idea in one paragraph

Faseeh never compares two voices. It compares two **phonetic scripts** (الخط العروضي) of the same
ayahs. The first is the reference: our code builds it from the Uthmani text with the tajweed rules of
Hafs by al-Shatibiyyah (`quran-transcript`), and it is the same for everyone, so it can be prepared in
advance. The second is the heard script: what the reader actually said, written letter by letter — and
only the AI model can produce it, because it depends on this person, today. Ordinary code then aligns
the two, and every difference that survives the filters in §4 becomes one mistake on one letter of one
word. The reference is the answer key; the model is the examiner. Without the model there is nothing to
grade.

---

## 2. Rules that must not be broken

These are not style choices. Breaking one makes the project wrong even if the code still runs.

| Rule | Why |
|---|---|
| Recitation is judged by **Hafs ʿan ʿĀṣim, ṭarīq al-Shāṭibiyyah** only | Agreed with the project's Qur'an specialist; other turuq (Ṭayyibah) have different madd lengths and are far outside the scope |
| Qur'anic text is the **Uthmani script of the Madinah Mushaf** (Tanzil), rendered in KFGQPC Uthmanic Script HAFS | The reader must read from the same text as the printed Mushaf; no simplification, no substitute font |
| **One** reference reciter: Muhammad Siddiq al-Minshawi (murattal) | The timings, the madd reference and the audio all assume a single reciter; mixing reciters breaks all three |
| The model is used **as published** — no fine-tuning, no retraining | Team decision; improvement happens through settings and calibration only (§4) |
| Analysis happens **after** the recording stops, never live | The whole server design (one request, one answer) depends on it |
| The **AI advisory notice** stays on the page, in every language | Required by the model's licence (QML 1.0). Removing it breaks the licence |
| The JavaScript classes at the top of `index.html` **match the UML class diagram exactly** | This is a Software Engineering course project; the diagram is the contract. Anything the UI needs beyond the diagram goes in a separate helper function, not as a new method on a model class |
| Every new interface string is translated into **all 13 languages** (`TEXT.xx`) | A missing translation shows an English string inside an Arabic or Urdu page |
| After editing `index.html` or `data/`, rebuild the single-file version | `python3 tools/build_standalone.py` → `standalone/Faseeh.html` |

---

## 3. How the parts fit together at runtime

```
                         GitHub, branch main  ──  the only source of truth
                                  │
          ┌───────────────────────┼────────────────────────────┐
          │                       │                            │
     GitHub Pages          Action: sync-hf-space.yml    team GPU machine
     faseeh.space          (on push to server/hf_space) (pulls every 2 min)
     the website                   │                            │
          │                        ▼                            ▼
          │            Hugging Face Space (CPU)        Docker + ngrok (GPU)
          │            always on, slower               fast, on only when started
          │                        │                            │
          └──── picks whichever answers /health, re-checked every minute ───┘
```

- **The website is static.** No backend, no database, no accounts. Everything it shows comes from
  `data/`, `audio/`, `images/` and the analysis server's answer.
- **The two servers run the same code** (`server/hf_space/`). The GPU machine is faster; the Space is
  the fallback that is always up. The page checks `/health` once a minute and uses whichever answers.
- **The GPU machine never starts by itself** — only `start_faseeh.bat`, until `stop_faseeh.bat`. It
  pulls the server code from GitHub every two minutes, so you deploy to it by pushing, not by copying.
  The exception is `server/local/` itself (the `.bat` files and compose files): those are copied by
  hand to the machine's folder.
- **URL modes.** `?server=<url>` uses another server for this visit only (https only, the reader is
  asked to confirm, and no name or team key is ever sent to it). `?server=` with nothing after it is
  the offline demo, which replays `data/ai-demo-response.json`. `?team` shows the server line and the
  review page link; `?team=0` turns it off.

### The analysis request

`POST /analyze` with the audio and header `X-Ayahs: 112:1,112:2,…`. The reader only chooses where to
**start**; the page always sends from the starting ayah to the end of the surah, and the server works
out how far they actually recited. The answer is text only — no HTML — and the page treats it as text:

```json
{ "errors": [ { "ayah": 1, "position": 2, "letter": "ه", "type": "letter_substitution" } ],
  "_meta": { "read": 4, "opening": 2.3, "words": { "1": [[0.42, 0.91], [0.91, 1.60], null] } } }
```

`read` = how many ayahs were recited, `opening` = seconds of isti'adha/basmala cut off the front,
`words` = where each word falls in the reader's own recording (used by the "You" playback buttons;
`null` means the word was not heard).

---

## 4. The knobs that decide what counts as a mistake

A correct reading flagged as wrong costs more than a real mistake missed. Everything here exists to
keep false alarms down.

**`server/hf_space/config.json`**

| Key | Now | Meaning |
|---|---|---|
| `votes` | `2` | The recording is heard three times (normal, 0.95×, 1.05×); a difference counts only if at least this many hearings agree |
| `min_conf` | `0.5` | A changed or added letter also needs this much model confidence |
| `calibration` / `calib_mode` | `true` / `"exact"` | Use `calibration.json`, matching the exact position |
| `madd_tolerance_counts` | `2` | Long madds (munfasil, muttasil, lazim) are compared **in letters of the phonetic script**, one letter per count, with this tolerance |
| `madd_tolerance_seconds` | `null` | The old method: comparing madd duration in seconds against al-Minshawi. Switched off deliberately — set a number here only to bring it back |

**`calibration.json`** — positions where expert reciters are flagged by mistake anyway, so they are
muted. Rebuilt with `server/calibrate.py` from correct recitations by six expert reciters. It is tied
to the settings above: change the settings and it should be rebuilt.

**`madd_reference.json`** — the reference madd lengths, built by `server/build_madd_reference.py`.
Only used when `madd_tolerance_seconds` is a number; without the file the server falls back to the
older behaviour.

**Environment variables** (Space secrets, or `settings.env` on the GPU machine):
`FASEEH_MAX_SECONDS` (330), `FASEEH_MAX_WAITING` (40 analyses at once, then HTTP 429),
`FASEEH_SAVES_PER_HOUR` / `FASEEH_SAVES_PER_READER`, `HF_TOKEN`, `FASEEH_DATASET`, `FASEEH_TEAM_KEY`.

Why a tolerance of two letters at all: strictly, a muttasil madd in Hafs needs at least four counts, so
two is an error. But the model writes fewer madd letters than an expert actually held when the expert
reads quickly. On Maher al-Muaiqly's correct reading of the 24 long madds in our seven surahs, a
tolerance of one letter produced 7 false alarms and a tolerance of two produced none. We chose two
with the project's Qur'an specialist: better to miss a shortened madd than to flag a correct reader.

---

## 5. Data files

| File | Holds | Watch out |
|---|---|---|
| `data/quran.json` | The seven surahs: Uthmani text and transliteration per ayah | Text is Tanzil's, unchanged except hamzah/maddah encoding |
| `data/reciters.json` | Reciter audio and the timing of **every word** | Generated by `server/align_reciter.py`, not taken from published timings (those ran ~0.1 s early). A surah whose alignment fails gets `exact: false` and is highlighted ayah by ayah instead of word by word |
| `data/articulation.json` | Articulation point of each letter: drawing + description in 13 languages | Drawings live in `images/` |
| `data/rules.json` | Tajweed rules and mistake explanations, 13 languages | Keyed by the Arabic rule name |
| `data/ai-demo-response.json` | A fixed answer for the offline demo | Only used when there is no server |

---

## 6. How to do the usual things

**Add a surah.** Add its text to `data/quran.json`; put the reciter's recording in `audio/`; run
`server/align_reciter.py` to generate its word timings into `data/reciters.json`; rebuild the
standalone file. Then re-run `calibrate.py`, because the muted positions are per ayah.

**Add or fix an interface language.** All strings live in the `TEXT` object in `index.html`, plus the
`description*` fields in `articulation.json` and the language keys in `rules.json`. Right-to-left
languages are listed in the layout code — Arabic, Urdu, Persian, Pashto and Punjabi.

**Change the reference reciter.** `server/prepare_reciter.py` (run on the GPU machine through
`prepare_reciter.bat`) produces the audio and the alignment; then `build_madd_reference.py`. Expect to
re-run `calibrate.py` and the evaluation afterwards: the reciter is baked into the madd reference and
the word timings.

**Re-tune the settings.** `evaluate.py sweep` on the tuning split, then `evaluate.py test` for the
honest numbers. Keep the splits apart — that is the whole point of them.

**Measure on real voices.** `test_site.bat` re-analyses the reviewed website recordings with today's
settings and times them. `evaluate.py own` uses your own recordings with `labels.csv`.

**Deploy.** Push to `main`. The website updates through GitHub Pages; anything under
`server/hf_space/` is pushed to the Space by the Action; the GPU machine picks it up within two
minutes. Wait for the green check in Actions, then hard-refresh.

---

## 7. What is proven, and what is not

Be careful with these numbers. They are honest, but they are not all the same kind of evidence.

| Measurement | Result | What it is |
|---|---|---|
| Mistake detection on qdat_bench | precision 96.2%, recall 89.3%, F1 92.7% | A public annotated benchmark of learner recordings |
| Correct expert readings passed with no false alarm | ~88% of 206 ayahs (ash-Shatri, Maher al-Muaiqly) | Studio recitations, not phone microphones |
| By kind of mistake (79 labelled recordings) | letters 98% / 93% · tajweed 97% / 90% · **short vowels 67% / 43%** | Precision / recall. Harakat are the weak point |
| First reviewed recordings from real users | precision 91%, recall 95% at word level (17 recordings) | Small sample |

**The caveat that matters:** these were measured before the madd rule changed from seconds to letter
counting. They should be measured again with today's settings.

Also untested on real voices at the time of writing: the isti'adha/basmala trimming and the
"server finds where you stopped" logic were verified against saved model output, not against fresh
recordings through a microphone.

---

## 8. Known limits

- **Short vowels (harakat) are the weakest part of the model.** Do not promise that Faseeh catches
  every tajweed ruling; it catches letters and words far better than fine rulings.
- **Calibration mutes real errors.** In the seven surahs it silences 88 tajweed positions (38 madd,
  27 ghunnah, 12 general tajweed, 7 qalqalah, 4 tarqiq). A genuine mistake at one of those positions
  is never reported. That is the price of not flagging correct readers.
- **Heavy/light letters (tafkhīm/tarqīq)** are only checked for ق ط خ, and only when a single ayah is
  analysed. The model returns ten attributes; we use two.
- **"Does not match the ayahs"** was returned wrongly for 5 of 22 real recitations in the first review
  round. The adaptive end-detection was written for exactly this and is believed to fix it, but that
  has not been confirmed on fresh recordings.
- **Saved recordings cannot be deleted on request.** There is no mechanism for it. Saving requires the
  reader's explicit consent and a name (or a generated anonymous one), goes to a private dataset, and
  is rate-limited — but it is not authenticated.
- Analysis time is not capped and grows with the length of the recording; it has never been measured
  formally.

---

## 9. Secrets and safety

- `settings.env` holds the GPU machine's ngrok token and the Hugging Face token. It is in
  `.gitignore` and **must never be committed** — nor anything else produced on that machine
  (`reviews.zip` contains voices and names).
- The Hugging Face token is fine-grained: write access to the Space and the recordings dataset only.
- The team key is sent in a header, never in the URL, and compared with `hmac.compare_digest`.
- Only the two official servers receive a reader's name, the team key or a request to save. Any other
  `?server=` address gets the audio and nothing else, over https, after the reader confirms.
- The Action that syncs the Space has read-only access to the repository.
- Still worth doing: two-factor authentication on the GitHub and Hugging Face accounts, and pinning
  the Docker image and library versions.

---

## 10. Where the rest of the documentation is

- **Requirements, system diagrams and the traceability matrix** — the project's requirements document
  (SRS), kept with the course submission.
- **How the AI model is used, with flowcharts** — the report *The Role of the AI Model in Faseeh*.
- **Diagrams** — `docs/diagrams/`: class, use case and behaviour diagrams.
- **Running the server on a machine with a graphics card** — `server/local/README.md`.
- **The model** — [Quran Muaalem](https://github.com/obadx/quran-muaalem) (`obadx/muaalem-model-v3_2`),
  arXiv 2509.00094, licence QML 1.0.

---

*Written at the end of the first phase of the project. If you change a rule in §2 or a number in §4,
please update this file in the same commit — it is only useful while it is true.*
