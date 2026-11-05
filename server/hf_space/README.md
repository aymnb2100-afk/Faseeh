---
title: Faseeh Server
emoji: 📖
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Faseeh analysis server

The AI server behind the Faseeh Qur'anic pronunciation corrector: `POST /analyze` with the recording
(header `X-Ayahs: 112:1,112:2`) returns the mistakes per ayah, word and letter. `GET /health` checks it is up.

Files: `faseeh_server.py`, `config.json` and `calibration.json` (from the evaluation), this `Dockerfile`.

Model: [Quran Muaalem](https://github.com/obadx/quran-muaalem) (arXiv 2509.00094), Quran Muaalem License (QML) 1.0.
The website using this server shows the licence's AI advisory notice to every user.
