"""
eeg_anonymization_real.py
====================================================================
Real-data counterpart of `eeg_anonymization.py`.

The original tutorial ran on a *synthetic* EEG dataset where we knew,
by construction, that identity lived in spectral-peak LOCATION and the
clinical task lived in alpha AMPLITUDE.  This script runs the SAME
anonymization methods and the SAME privacy/utility evaluation on a real
recording set: the PhysioNet "Auditory evoked potential EEG biometric"
dataset stored in ./Raw_Data/.

What changed vs. the synthetic version
--------------------------------------
  * ACT 1 (synthesise)  ->  load_dataset():  parse the OpenBCI .txt
                            files, slice them into fixed-length epochs.
  * FS 128 -> 200 Hz, 4 EEG channels instead of 1.
  * Feature extraction works per channel, then concatenates channels.
  * Labels come from the file name:
        s##  -> SUBJECT  = identity      (privacy / re-ID target)
        ex## -> EXERCISE = condition     (utility / task target)
    This is a biometric dataset, so subject re-identification is a
    genuine privacy threat, not a synthetic one.

What stayed the same
--------------------
  * ACT 2 (evaluate): a re-ID attacker + a task decoder, both retrained
    on the released features (honest, adaptive threat model).
  * ACT 3 (anon_*): every anonymisation transform f(.), generalised to
    operate channel-wise.
  * ACT 4: the privacy-utility plane and before/after figures.

Data format (per file, comma-separated, one header row)
    Sample Index, EXG Channel 0..3, Accel 0..2, Other x5, Timestamp,
    Timestamp (Formatted)
    -> we keep EXG Channel 0..3 (columns 1..4), sampled at ~200 Hz.
====================================================================
"""

import os
import glob
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# Recording parameters (measured from the Raw_Data files)
# --------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Raw_Data")
FS = 200                  # sampling rate (Hz), measured from timestamps
EPOCH_SEC = 4             # length of one analysis window (s)
N_SAMPLES = FS * EPOCH_SEC          # 800 samples per epoch
NFREQ = N_SAMPLES // 2 + 1          # rfft output length
FREQS = np.fft.rfftfreq(N_SAMPLES, 1 / FS)   # frequency axis (Hz)

EXG_COLS = [1, 2, 3, 4]   # the 4 EEG channels in each .txt row
N_CHANNELS = len(EXG_COLS)

# Frequency band edges (Hz) used for coarse band-power features
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# We analyse 1-40 Hz; below 1 Hz is drift, above 40 is mains/EMG.
FMASK = (FREQS >= 1) & (FREQS <= 40)

# How much of each (long) recording to use, to keep runtime sane.
MAX_EPOCHS_PER_FILE = 20            # 20 epochs * 4 s = 80 s per file
WINDOW = np.hanning(N_SAMPLES)      # taper to curb spectral leakage


# ====================================================================
# ACT 1 (replaced) -- LOAD A LABELLED EEG DATASET FROM Raw_Data/
# ====================================================================
_FNAME_RE = re.compile(r"^(s\d+)_(ex\d+)")


def _read_recording(path):
    """Read the 4 EEG channels from one OpenBCI .txt file.

    Returns an array of shape (n_channels, n_timepoints). Duplicate
    sample-index rows in the file are genuine consecutive samples (the
    index is just OpenBCI's 0-255 counter), so we keep every data row.
    """
    with warnings.catch_warnings():            # ignore truncated trailing rows
        warnings.simplefilter("ignore")
        raw = np.genfromtxt(path, delimiter=",", skip_header=1,
                            usecols=EXG_COLS, invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]      # drop malformed rows
    return raw.T                               # (n_channels, n_timepoints)


def _epoch(sig):
    """Slice (n_channels, T) into non-overlapping epochs.

    Each epoch is linearly detrended (kills the large EEG DC offset and
    slow drift) and Hann-windowed, then returned as
    (n_epochs, n_channels, N_SAMPLES).
    """
    n = sig.shape[1] // N_SAMPLES
    n = min(n, MAX_EPOCHS_PER_FILE)
    if n == 0:
        return np.empty((0, N_CHANNELS, N_SAMPLES))
    chunks = sig[:, : n * N_SAMPLES].reshape(N_CHANNELS, n, N_SAMPLES)
    chunks = np.transpose(chunks, (1, 0, 2))   # (n_epochs, n_ch, N)
    chunks = detrend(chunks, axis=-1, type="linear")
    return chunks * WINDOW


def load_dataset(data_dir=DATA_DIR, subjects=None, exercises=None):
    """Build (X, y_subj, y_task, subj_names, task_names) from the files.

    X         : (n_epochs, n_channels, N_SAMPLES) time-domain epochs
    y_subj    : integer subject id per epoch          (identity label)
    y_task    : integer exercise id per epoch         (condition label)
    subj_names/task_names: the string labels behind those integers.

    `subjects` / `exercises` optionally restrict the load, e.g.
    exercises=["ex01", "ex02"] to study a clean two-condition contrast.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "s*_ex*.txt")))
    X, subj_lbl, task_lbl = [], [], []
    for path in paths:
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        subj, exer = m.group(1), m.group(2)
        if subjects and subj not in subjects:
            continue
        if exercises and exer not in exercises:
            continue
        epochs = _epoch(_read_recording(path))
        if len(epochs) == 0:
            continue
        X.append(epochs)
        subj_lbl += [subj] * len(epochs)
        task_lbl += [exer] * len(epochs)

    if not X:
        raise RuntimeError(f"No usable recordings found in {data_dir!r}")

    X = np.concatenate(X, axis=0)
    subj_names = sorted(set(subj_lbl))
    task_names = sorted(set(task_lbl))
    s_idx = {s: i for i, s in enumerate(subj_names)}
    t_idx = {t: i for i, t in enumerate(task_names)}
    y_subj = np.array([s_idx[s] for s in subj_lbl])
    y_task = np.array([t_idx[t] for t in task_lbl])
    return X, y_subj, y_task, subj_names, task_names


# ====================================================================
# Feature extraction -- the "released representation" (per channel)
# ====================================================================
def _psd_3d(X):
    """Per-channel PSD, kept on the 1-40 Hz grid.

    X    : (n_epochs, n_channels, N_SAMPLES)
    out  : (n_epochs, n_channels, n_freq_in_band)
    """
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2 / N_SAMPLES
    return P[..., FMASK]


def _flat(feat3d):
    """Concatenate the per-channel feature axis into one vector/epoch."""
    return feat3d.reshape(feat3d.shape[0], -1)


def psd(X):
    """2-D released spectral view (channels concatenated)."""
    return _flat(_psd_3d(X))


def log_psd(X):
    return _flat(np.log(_psd_3d(X) + 1e-12))


def band_powers(X):
    """Coarse summary: log power in each classic band, per channel.

    -> (n_epochs, n_channels * 4) features.
    """
    P = _psd_3d(X)                     # (n_ep, n_ch, n_freq)
    f = FREQS[FMASK]
    out = []
    for lo, hi in BANDS.values():
        out.append(P[..., (f >= lo) & (f < hi)].sum(axis=-1))   # (n_ep, n_ch)
    return np.log(np.stack(out, axis=-1).reshape(P.shape[0], -1) + 1e-12)


# ====================================================================
# ACT 2 -- THE TWO EVALUATORS  (unchanged from the synthetic version)
# ====================================================================
def evaluate(features, y_subj, y_task, seed=0):
    """Train+test a re-ID attacker and a task decoder on the SAME
    released features. Returns (reid_accuracy, task_accuracy).

    Both models are retrained on the anonymised features -- the honest
    threat model: the attacker adapts to your defence.
    """
    idx = np.arange(len(y_subj))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                              stratify=y_subj)

    # Privacy: can the attacker name the subject? (chance = 1/n_subjects)
    reid = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    reid.fit(features[tr], y_subj[tr])
    reid_acc = reid.score(features[te], y_subj[te])

    # Utility: can the decoder still read the condition? (chance = 1/n_task)
    task = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    task.fit(features[tr], y_task[tr])
    task_acc = task.score(features[te], y_task[te])

    return reid_acc, task_acc


# ====================================================================
# ACT 3 -- THE ANONYMISATION TRANSFORMS f(.)  (generalised to channels)
# Each returns the released FEATURE matrix for a given raw X.
# ====================================================================
def anon_raw(X):
    """No anonymisation -- baseline. High utility AND high identity."""
    return log_psd(X)


def anon_gaussian_noise(X, snr_db, rng):
    """Naive perturbation: add white noise at a target SNR (time domain)."""
    p_sig = np.mean(X ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noisy = X + np.sqrt(p_noise) * rng.standard_normal(X.shape)
    return log_psd(noisy)


def anon_dp_features(X, epsilon, rng, clip=8.0):
    """Differential-privacy-style Laplace mechanism on the features.

    Clip features to a bounded range so sensitivity is finite, then add
    Laplace noise scaled by sensitivity/epsilon. Smaller epsilon = more
    privacy = more noise. (A teaching-grade DP demo, not an audited one.)
    """
    feats = np.clip(log_psd(X), -clip, clip)
    sensitivity = 2 * clip
    scale = sensitivity / epsilon
    return feats + rng.laplace(0.0, scale, feats.shape)


def anon_spectral_smooth(X, sigma_hz):
    """Blur the spectrum along frequency (per channel).

    Washes out the precise LOCATION of each subject's peaks (identity)
    while preserving integrated band power (the condition signal).
    """
    df = FREQS[1] - FREQS[0]
    smoothed = gaussian_filter1d(np.log(_psd_3d(X) + 1e-12),
                                 sigma=sigma_hz / df, axis=-1)
    return _flat(smoothed)


def anon_band_powers(X):
    """Collapse to band powers only (per channel).

    Throws away fine spectral location (kills the frequency fingerprint)
    but keeps band power, so the condition can survive.
    """
    return band_powers(X)


def anon_iaf_align(X):
    """Frequency-warp each channel so its alpha peak sits at 10 Hz.

    Removes the individual-alpha-frequency biometric while preserving
    alpha magnitude. On real data the alpha peak is not always crisp, so
    this is a realistic 'partial' defence.
    """
    P = np.log(_psd_3d(X) + 1e-12)          # (n_ep, n_ch, n_freq)
    f = FREQS[FMASK]
    alpha_mask = (f >= 8) & (f <= 13)
    alpha_bins = np.where(alpha_mask)[0]
    ref_bin = np.argmin(np.abs(f - 10.0))
    aligned = np.empty_like(P)
    for i in range(P.shape[0]):
        for c in range(P.shape[1]):
            row = P[i, c]
            peak_bin = alpha_bins[np.argmax(row[alpha_mask])]
            aligned[i, c] = np.roll(row, ref_bin - peak_bin)
    return _flat(aligned)


def anon_phase_randomize(X, rng):
    """Surrogate-data trick: randomise phases, keep the magnitude
    spectrum (per channel). Looks like anonymisation but PRESERVES the
    power spectrum -- which is exactly the fingerprint. Included to show
    a method that FAILS."""
    spec = np.fft.rfft(X, axis=-1)
    phases = rng.uniform(0, 2 * np.pi, spec.shape)
    surrogate = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=N_SAMPLES)
    return log_psd(surrogate)


# ====================================================================
# ACT 4 -- RUN THE EXPERIMENT
# ====================================================================
def main():
    rng = np.random.default_rng(42)

    # Load everything. To study a clean two-condition contrast instead,
    # pass e.g. exercises=["ex01", "ex02"].
    X, y_subj, y_task, subj_names, task_names = load_dataset()
    n_subjects = len(subj_names)
    n_tasks = len(task_names)
    chance_reid = 1.0 / n_subjects
    chance_task = 1.0 / n_tasks

    print(f"Dataset: {X.shape[0]} epochs x {X.shape[1]} channels x "
          f"{X.shape[2]} samples")
    print(f"  {n_subjects} subjects: {', '.join(subj_names)}")
    print(f"  {n_tasks} conditions: {', '.join(task_names)}")

    # ---- single-point methods --------------------------------------
    results = {}
    results["Raw (no anonymisation)"] = evaluate(anon_raw(X), y_subj, y_task)
    results["Phase randomise"]        = evaluate(anon_phase_randomize(X, rng), y_subj, y_task)
    results["IAF alignment"]          = evaluate(anon_iaf_align(X), y_subj, y_task)
    results["Spectral smooth (3 Hz)"] = evaluate(anon_spectral_smooth(X, 3.0), y_subj, y_task)
    results["Band powers only"]       = evaluate(anon_band_powers(X), y_subj, y_task)

    # ---- swept methods (draw a curve) ------------------------------
    noise_curve, dp_curve, smooth_curve = [], [], []
    for snr in [20, 12, 6, 0, -4, -8]:
        noise_curve.append((snr,) + evaluate(anon_gaussian_noise(X, snr, rng), y_subj, y_task))
    for eps in [50, 20, 10, 5, 2, 1]:
        dp_curve.append((eps,) + evaluate(anon_dp_features(X, eps, rng), y_subj, y_task))
    for sig in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        smooth_curve.append((sig,) + evaluate(anon_spectral_smooth(X, sig), y_subj, y_task))

    # ---- report ----------------------------------------------------
    print(f"\nChance levels:  re-ID = {chance_reid:.3f}   task = {chance_task:.3f}")
    print(f"{'method':<26}{'re-ID (privacy)':>16}{'task (utility)':>16}")
    print("-" * 58)
    for name, (r, t) in results.items():
        print(f"{name:<26}{r:>16.3f}{t:>16.3f}")

    # ================================================================
    # FIGURE 1 -- see the data: two subjects' mean spectra (channel 0)
    # ================================================================
    f = FREQS[FMASK]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for s, color in [(0, "#534AB7"), (min(5, n_subjects - 1), "#D85A30")]:
        m = y_subj == s
        ax[0].plot(f, _psd_3d(X[m])[:, 0].mean(0), color=color, lw=2,
                   label=subj_names[s])
    ax[0].set_title("Per-subject mean spectrum (channel 0)\nidentity cues live here")
    ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Power")
    ax[0].legend(); ax[0].set_xlim(1, 40)

    for t, color in [(0, "#0F6E56"), (min(1, n_tasks - 1), "#185FA5")]:
        m = y_task == t
        ax[1].plot(f, _psd_3d(X[m])[:, 0].mean(0), color=color, lw=2,
                   label=task_names[t])
    ax[1].axvspan(8, 13, color="0.85", alpha=0.5, label="alpha band")
    ax[1].set_title("Per-condition mean spectrum (channel 0)\nutility cues live here")
    ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("Power")
    ax[1].legend(); ax[1].set_xlim(1, 40)
    fig.tight_layout(); fig.savefig("fig1_signal_structure_real.png", dpi=130)

    # ================================================================
    # FIGURE 2 -- the privacy-utility plane (the money plot)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.axhline(chance_task, ls=":", c="grey"); ax.axvline(chance_reid, ls=":", c="grey")
    ax.text(chance_reid + .01, chance_task + .02, "re-ID chance", color="grey", fontsize=9)
    ax.text(.78, chance_task + .005, "task chance", color="grey", fontsize=9)

    for (name, (r, t)), mk in zip(results.items(), ["*", "X", "P", "s", "D"]):
        ax.scatter(r, t, s=80, marker=mk, zorder=5, label=name)

    def plot_curve(curve, color, label):
        c = np.array([(r, t) for _, r, t in curve])
        ax.plot(c[:, 0], c[:, 1], "-o", color=color, alpha=.8, label=label, ms=4)

    plot_curve(noise_curve, "#BA7517", "additive noise (sweep SNR)")
    plot_curve(dp_curve, "#993556", "DP Laplace (sweep epsilon)")
    plot_curve(smooth_curve, "#185FA5", "spectral smooth (sweep sigma)")

    ax.set_xlabel("Re-identification accuracy")
    ax.set_ylabel("Task accuracy")
    ax.set_title("The privacy-utility plane for real EEG anonymisation")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(min(0.0, chance_task - .05), 1.02)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); fig.savefig("fig2_privacy_utility_real.png", dpi=130)

    # ================================================================
    # FIGURE 3 -- before/after a good transform (spectral smoothing)
    # ================================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    before = np.log(_psd_3d(X) + 1e-12)[:, 0]                  # channel 0
    after = anon_spectral_smooth(X, 3.0).reshape(
        X.shape[0], N_CHANNELS, -1)[:, 0]
    for s, color in [(0, "#534AB7"), (min(5, n_subjects - 1), "#D85A30")]:
        m = y_subj == s
        ax[0].plot(f, before[m].mean(0), color=color, lw=2, label=subj_names[s])
        ax[1].plot(f, after[m].mean(0), color=color, lw=2, label=subj_names[s])
    ax[0].set_title("Before: subjects separable by spectral detail")
    ax[1].set_title("After 3 Hz smoothing: fine peaks erased,\nband power preserved")
    for a in ax:
        a.axvspan(8, 13, color="0.85", alpha=0.5); a.set_xlabel("Frequency (Hz)")
        a.set_xlim(1, 40); a.legend()
    ax[0].set_ylabel("log power")
    fig.tight_layout(); fig.savefig("fig3_before_after_real.png", dpi=130)

    print("\nSaved: fig1_signal_structure_real.png, "
          "fig2_privacy_utility_real.png, fig3_before_after_real.png")


if __name__ == "__main__":
    main()
