# فصيح · Faseeh

**مصحّح النطق القرآني** — a web app that helps readers correct their Qur'anic pronunciation.
The reader picks a surah, recites it, and sees each mispronounced letter with its phonetic
transcription, its articulation point and the reciter's audio. Any word can also be studied
directly, without recording.

Built for a Software Engineering course: the JavaScript classes in `index.html` follow the
project's UML class diagram one to one (see [Architecture](#architecture)).

> **AI analysis:** recordings are analysed by `server/faseeh_server.py`, which runs the
> [Quran Muaalem](https://github.com/obadx/quran-muaalem) model (Wav2Vec2-BERT, multi-level CTC, trained on
> about 850 hours of Hafs recitation; arXiv 2509.00094). Without a server the page shows a fixed sample
> analysis from `data/ai-demo-response.json`.

---

## Features

- **Mushaf-style page**: Uthmani Ḥafṣ script, ornamental frame, surah title in calligraphy.
- **Ayah selection (optional)**: tap a first and last ayah, or recite the whole surah.
- **Hear the reciter**: Sheikh Muhammad Siddiq al-Minshawi (murattal) recites the selected ayahs (or the whole surah)
  while the current ayah and word are highlighted on the page.
- **Recording**: microphone, or an uploaded audio file when no microphone is available.
- **My mistakes**: wrong words turn red; each card shows
  - the phonetic transcription,
  - the articulation point (drawing + description),
  - the expert recitation by Muhammad Siddiq al-Minshawi, for the word alone or the whole ayah, and your own recording.
  Several wrong letters in one word get one tab per letter.
- **Word pronunciation**: select an ayah, pick any word and letter, and see the same card plus
  the tajweed rules for that letter (qalqalah, ghunnah, madd, nūn/mīm sākinah, lām shamsiyyah…).
- **Show** toggles: choose which parts of the card to display.
- **Advanced rules**: adds the tajweed rule under the card.
- **Transliteration** line, one ayah at a time.
- **Interface in 13 languages**: العربية, English, Español, Français, 中文, اردو, Bahasa Indonesia, বাংলা,
  پنجابی (Shahmukhi), فارسی, Türkçe, پښتو, Hausa — including the tajweed rules and articulation descriptions.
  Arabic, Urdu, Persian, Pashto and Punjabi switch the page to right-to-left. The menu remembers the choice.
- **Light / dark** mode and **display size** (85–150%).
- **How to use** card and short tooltips on controls that need explaining.
- Works on phones, tablets and desktops.

Surahs included: النبأ (78), عبس (80), العصر (103), الكوثر (108), الإخلاص (112), الفلق (113), الناس (114).

---

## Run it

### Option 1: open the single file
Open **`standalone/Faseeh.html`** in a browser (double-click). Everything, including the data,
is inside that one file.

### Option 2: run the project folder
`index.html` loads its data with `fetch()`, which browsers block for files opened directly from
disk, so serve the folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

### Option 3: GitHub Pages
1. Push this folder to a GitHub repository.
2. **Settings → Pages → Build and deployment**: Source *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site appears at `https://<your-user>.github.io/<repo-name>/`.

An internet connection is needed for the interface font (Google Fonts) and the reciter audio. The Qur'anic fonts are embedded in the page.

### Rebuild the single file
After editing `index.html` or anything in `data/`:

```bash
python3 tools/build_standalone.py
```

---

## AI server

The quickest way is Google Colab (free GPU): open `server/Faseeh_Server_Colab.ipynb`, choose a T4 GPU and
run the cells. The notebook starts the server, prints a public link, and runs the evaluation.

Then open the website once with `?server=<that link>` (the address is remembered in the browser), or put it in
`DEFAULT_SERVER` in `index.html` for everyone. `?server=` with nothing after it returns to the demo.

How a recording is analysed:
1. the browser sends the audio and the selected ayahs (`POST /analyze`, header `X-Ayahs: 112:1,112:2`);
2. the model writes down the phonemes the reader actually said, with their articulation attributes (sifat);
3. `quran-transcript` produces the correct Hafs phonemes for those ayahs;
4. the two are aligned; every difference becomes a mistake on a word and letter: letter replaced or left out,
   wrong vowel, madd length, ghunnah, qalqalah, heavy letters read light.

The model's licence (QML 1.0) requires showing its AI advisory notice to every user, in every interface
language; the page shows it before the first recording and at the bottom of the page.

### Avoiding false alarms
A correct reading must not be marked wrong, so the server only reports a mistake when
1. it is heard in at least 2 of 3 versions of the recording (normal speed, 5% slower, 5% faster),
2. the model is confident enough about a replaced or added sound, and
3. the model does not "hear" the same mistake on the same word when expert reciters read that ayah correctly
   (`calibrate.py` → `calibration.json`); the madd lengths allowed in Hafs are all accepted.

### Evaluation
`server/evaluate.py` (also run from the notebook). The data is split so the test numbers are honest:

| Part | Data | Used for |
|---|---|---|
| calibration | Husary, Minshawy, Abdul Basit, Sudais, Shuraym, Ghamdi (EveryAyah) | the model's habitual slips (`calibrate.py`) |
| tuning | Hudhaify, Muhammad Ayyoub + even-numbered qdat_bench samples | `evaluate.py sweep` chooses the settings (`config.json`) |
| test | Abu Bakr ash-Shatri, Maher al-Muaiqly + odd-numbered qdat_bench samples | `evaluate.py test`: the reported numbers |
| own | your recordings with known mistakes (`own_recordings/labels.csv`) | `evaluate.py own` |

Metrics: share of correct expert ayahs passed with no false alarm; precision / recall / F1 of mistake detection
on learner recordings (qdat_bench, 159 annotated recordings); phoneme error rate.

---|---|---|
| benchmark | qdat_bench: 159 annotated learner recordings | phoneme error rate; mistake detection precision / recall / F1 |
| experts | Abu Bakr ash-Shatri and Maher al-Muaiqly reading every ayah of the 7 surahs | share of correct ayahs passed with no false alarm |
| own | your own recordings with known mistakes (`own_recordings/labels.csv`) | word-level precision / recall / F1 |

---

## Project structure

```
index.html                  The whole app: HTML, CSS and JavaScript (fonts embedded)
data/
  quran.json                Surahs: Uthmani text and transliteration per ayah
  articulation.json         Articulation point of each letter (drawing + description in 13 languages)
  rules.json                Tajweed rules and mistake explanations (13 languages)
  reciters.json             Reciter audio: one recording per surah with the timing of every word
  ai-demo-response.json     Sample AI response used by the demo AIEngine
images/                     Articulation drawings, one per letter
standalone/Faseeh.html      Single-file build (generated)
tools/build_standalone.py   Builds standalone/Faseeh.html
server/
  faseeh_server.py          AI analysis server (FastAPI + Quran Muaalem)
  calibrate.py              Finds the model's habitual slips on expert recitations
  evaluate.py               Evaluation: tuning (sweep), test, own recordings
  Faseeh_Server_Colab.ipynb Runs the server and the evaluation on Google Colab
docs/diagrams/
  class/                    Class diagram: PlantUML, PNG, SVG, draw.io, course-tool JSON
  use-case/                 Use case diagram: PlantUML, PNG, draw.io
  behavior/                 Sequence (SD1–SD4) and activity (AD1–AD6) diagrams
```

---

## Architecture

The domain classes live at the top of the `<script>` in `index.html`, translated from the class
diagram (`docs/diagrams/class/Faseeh_Final.puml`):

| Class | Role |
|---|---|
| `Surah` ◆— `Ayah` | Qur'an text, loaded from `data/quran.json` (composition) |
| `User` | Language, transliteration and advanced-rules settings; creates recordings |
| `AudioRecording` | Microphone permission, record/stop, playback |
| `AIEngine` | Sends a recording for analysis and returns `PronunciationError`s |
| `PronunciationError` | One wrong letter in one word of an ayah |
| `CorrectionInterface` «GUI» ◆— `PhoneticTranscription`, `ArticulationGuide`, `ExpertAudio` | The correction card |

UML visibility maps to JavaScript as: private `-` → `#field`, public `+` → method.
Display preferences (theme, display size, which card parts are shown) are stored in the browser
and are not part of the domain model.

> **Diagrams vs. code:** the class diagram is up to date with the code. It includes
> `CorrectionInterface.loadWordDetails(ayah, positionIndex, letter)` and `ruleKeys` (Word pronunciation),
> `ExpertAudio.wordAudioPath` with `getWordAudioClip()`, and multiplicity `0..1` on the
> `PronunciationError` side of *displaysIn* (a card can show a looked-up word with no error).
> The use case and behaviour diagrams still predate *Look up word pronunciation* and *Listen to reciter*.

---

## Credits

- **Qur'an text**: Tanzil Project (Uthmani, Ḥafṣ), via
  [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api). Only the character
  encoding of hamzah and maddah was normalised; the text is unchanged.
- **Qur'anic font**: KFGQPC Uthmanic Script HAFS, King Fahd Glorious Qur'an Printing Complex.
- **Surah name font**: from [quran.com-frontend-next](https://github.com/quran/quran.com-frontend-next).
- **Wordmark font**: Mirza (SIL Open Font License 1.1).
- **Interface fonts**: IBM Plex Sans Arabic, with Amiri / Amiri Quran as fallbacks, via Google Fonts (SIL Open Font License 1.1).
- **Pronunciation model and phonetic script**: Quran Muaalem and quran-transcript by obadx
  ([github.com/obadx/quran-muaalem](https://github.com/obadx/quran-muaalem)), Quran Muaalem License (QML) 1.0.
- **Recitation audio**: Muhammad Siddiq al-Minshawi (murattal). Surah recordings from the Quran.com audio service
  (`download.quranicaudio.com`, recitation id 9), word timings aligned by Faseeh; per-ayah fallback from [everyayah.com](https://everyayah.com).
