# Dataset

The recordings are **not distributed** with this repository (they show a person's face and voice).
This page documents what they are so the code and results can be understood and reproduced on
similar data.

## Recordings

| | |
|---|---|
| Task | speaker-dependent isolated-word recognition, 20 first names |
| Vocabulary | Ahmed, Amber, Charlie, Christopher, Dominic, Emad, Emma, Hannah, Imogen, Jess, Josh, Joshua, Kailong, Kira, Manwel, Mateusz, Ngozi, Riley, Sivaprasath, Zack |
| Audio-visual session | 600 clips = 20 names × 30 takes, one speaker, smartphone, 1280×720 H.264, AAC 48 kHz stereo, ~3.1–3.7 s per clip |
| Audio-only session | 400 clips = 20 × 20 takes recorded about a month earlier with a different microphone, 16 kHz mono (used only as an out-of-session test set) |
| Room noise | 22 s recording of the recording room (used as the "ambient" noise type) |
| Recorded by | Hanming Zhang (training and test recordings) |

Some vocabulary pairs are deliberately confusable (Josh / Joshua, Emma / Imogen / Hannah) and some
words are very short (Jess, Zack, Kira).

## Properties found while building the pipeline

These are measured by `avsr prepare` and the analysis code, not assumed:

* **Variable frame rate.** Nominal 60 fps, actual ~29–31 fps with irregular gaps (e.g. a 96 ms gap
  at the start of a clip). OpenCV's per-frame timestamps were checked against `ffprobe` (max
  difference < 0.001 ms) and every visual feature keeps its timestamp.
* **Audio starts late.** The AAC stream's start time is 89–396 ms (mean 272 ms) after the first
  video frame. Extracting audio naively drops this offset; the pipeline re-inserts it.
* **Leading digital silence.** In ~90% of clips the audio is exactly zero until shortly before the
  word.
* **Block-wise recording.** Container timestamps show that the 30 takes of each name were
  recorded consecutively (e.g. all "Ahmed" takes within four minutes, "Zack" about 2.5 hours
  after "Ahmed"). Anything that drifts slowly — lighting, head pose, camera placement — is
  therefore correlated with the label. See the confound analysis in the technical report.

## Expected layout

```
<video_root>/<Name>/<NameNNN>.mp4          # one folder per name (the 'Ngozi' folder is named 'Ngoz')
<session2_audio_root>/<Name>/<NameNNN>.wav
<ambient_noise>.wav                         # 16 kHz mono
```

Paths are set in `configs/default.yaml` or with the environment variables `AVSR_VIDEO_ROOT`,
`AVSR_SESSION2_ROOT` and `AVSR_AMBIENT_NOISE`.

## Split

Fixed stratified split, identical for every experiment: per name 18 takes for training, 6 for
validation (model selection, early stopping, fusion weights) and 6 for testing (seed 2025, see
`avsr/data/manifest.py`).
