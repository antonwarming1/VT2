# Project Problems & Design Decisions

## 1. Does resampling (Step 2) make sense?

**Yes, resampling is necessary and correct — but it introduces a subtle duration-leakage problem.**

### Why resampling is needed
- CSV (Task data) arrives at **irregular intervals of ~2–36 ms**.
- JSON (Intrinsic data) arrives at a **uniform 1 ms**.
- tsfresh features like FFT coefficients, autocorrelation, and spectral coefficients **assume uniform time spacing**. If the input is irregular, these features are mathematically wrong (e.g. FFT coefficient 76 doesn't correspond to a fixed frequency unless samples are evenly spaced).
- Resampling to 2 ms aligns both sources on the same time grid so that joint features are comparable across samples.

### The duration-leakage problem this creates

After resampling at 2 ms, the number of rows in a series is:

```
n_samples = duration_ms / 2
```

tsfresh extracts a `length` feature (number of samples), and **all three signal channels passed feature selection with a `length` feature**:

- `intr_Current (V)__length`
- `intr_Torque (Nm)__length`
- `task_Robot_I (A)__length`

This means the classifier can tell classes apart partly by how long the screwing took. Whether this is "cheating" depends on the use case:

| Situation | Verdict |
|-----------|---------|
| Classes physically differ in duration (OT completes faster, UT never reaches target, NS barely engages) | **Legitimate** — duration is a real signal |
| Goal is to be robust to speed/torque variations or different operators | **Leakage** — model may not generalise |
| Deploying in real-time before the screw finishes | **Unusable** — duration is not known yet |

**For old data specifically**: data is trimmed at the depth plateau (`detect_plateau`), so the series spans from screwing-start to the moment depth stops changing. Different fault types genuinely reach different plateaus at different times, making `length` a physically grounded feature.

---

## 2. Does Savitzky-Golay smoothing (Step 3) make sense?

**Yes, but with important caveats.**

### Why it makes sense
- `Robot_I (A)` (motor current from robot controller) has electrical switching noise that is not physically meaningful.
- Savitzky-Golay with window=11, poly=3 fits a polynomial through 11 consecutive points and evaluates at the centre — it preserves peaks and slopes much better than a moving average, which is important for screwing events (torque spikes, current surges) that are genuinely sharp.
- Smoothing is applied **after resampling**, which is correct: SavGol assumes uniform spacing, so it must run on the already-uniformly-resampled data.

### Potential issues
- `SMOOTH_INTR = True` also smooths `Torque (Nm)` and `Current (V)` from the intrinsic JSON. Torque in particular may have **real high-frequency content** (oscillations during thread engagement, strip events) that smoothing could destroy. If FFT features from Torque are important, verify that the signal content above ~45 Hz (window=11 at 2 ms = 22 ms span ≈ 45 Hz cutoff) is noise and not signal.
- The current `SAVGOL_WINDOW = 11` is the same for all columns. A wider window is safer for Robot_I (slow-varying current envelope) but may be too aggressive for Torque (fast-varying screw dynamics).

---

## 3. WAV files (44100 Hz acoustic data) are unused

**The pipeline ignores all `.wav` files. This is a significant gap.**

### What exists
WAV files are stored alongside CSV/JSON pairs in `data/` (e.g. `020320261A1.wav`). The old dataset (`Data fra tidligere project/Dataset/Extrinsic data (clean)/`) also has WAVs. These are acoustic emission recordings of the screwing process.

### Why they cannot just be added to the current pipeline
- The current resampling target is **2 ms (500 Hz)**.
- WAV files are at **44100 Hz** — 88× higher temporal resolution.
- Downsampling a 44100 Hz signal to 500 Hz loses all the acoustic content; the entire value of audio is in the mid-to-high frequency range (screw thread noise, material cracking, slippage).
- They cannot be treated as another CSV column and fed through the same `resample_uniform` → tsfresh pipeline.

### What would be needed
Audio features must be extracted separately, typically:
- **MFCCs** (Mel-frequency cepstral coefficients) — standard for audio classification
- **Spectral centroid, roll-off, bandwidth** — characterise the sound texture
- **Short-time FFT / spectrogram** — preserves time-frequency structure

These are then concatenated with the CSV/JSON tsfresh features at the sample level (one row per screw, not per time step).

### Open question
Are the WAV files time-synchronised with the CSV/JSON data? If yes, the audio likely captures the exact same screwing event and is highly informative. If not, synchronisation must be established before using them.

---

## 4. Does including Time (ms) cause duration leakage?

**Partially correct concern, but the mechanism is different from what you might think.**

### How tsfresh actually handles Time (ms)
In `Feature_engineering/code.py`:

```python
cols = [c for c in df.columns if c != "Time (ms)"]   # excludes Time from features
df = df.rename(columns={"Time (ms)": "time"}).copy()
return df[["id", "time"] + cols]
```

Then in tsfresh:
```python
extract_features(df, column_id="id", column_sort="time", ...)
```

`column_sort="time"` tells tsfresh to order rows by time but **not to extract features from the time column itself**. The actual millisecond values never enter any feature calculation — tsfresh treats the sorted series as index-based (sample 0, 1, 2, …).

### Where the real leakage comes from
After uniform resampling, **series length encodes duration**. tsfresh's `length` feature counts rows. Since all three channels have `length` in `features_selected.csv` and they passed statistical feature selection, the classifier is already using duration.

This is not caused by Time (ms) being a column — it would happen regardless, as long as different classes have different numbers of rows.

### Does it matter?
For this classification task (fault type in screwing), duration is probably a legitimate physical signal. However, it is worth verifying: if you trained a trivial classifier using only the `length` features (i.e., just duration), what accuracy would you get? If it is already high (> 70%), the model may be relying more on duration than on signal shape.

---

## Summary table

| Issue | Severity | Action |
|-------|----------|--------|
| Resampling (Step 2) is needed for FFT/spectral tsfresh features | — | No change needed |
| `length` feature encodes duration, passed feature selection | Medium | Check if a duration-only baseline is already accurate; decide if duration should be excluded |
| SavGol on Robot_I (Step 3) | — | Fine as-is |
| SavGol on Torque may kill real signal content | Low–Medium | Optionally try `SMOOTH_INTR = False` and compare FFT features |
| WAV files (44100 Hz) are completely unused | High | Needs a separate audio feature extraction pipeline |
| Time (ms) as column does NOT cause leakage via tsfresh | — | Misunderstanding; no action needed on Time (ms) itself |
