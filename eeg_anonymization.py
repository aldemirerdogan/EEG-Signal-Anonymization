"""
eeg_anonymization.py
====================================================================
A from-scratch, self-contained tutorial on signal-level anonymization
of a 1-D medical signal (EEG).

The script is organised as a small experiment in four acts:

    1. SYNTHESISE   a labelled EEG dataset where we *control* what is
                    "identity" and what is "clinical task" information.
    2. ATTACK       build a re-identification attacker (privacy metric)
                    and a task decoder (utility metric).
    3. ANONYMISE    apply several transforms f(.) of increasing
                    sophistication.
    4. EVALUATE     plot every method on the privacy-utility plane so
                    the trade-off is visible, not asserted.

Why synthetic data?  Because then we KNOW the ground truth: identity
lives in the *frequency location* of each subject's spectral peaks
(their individual alpha frequency, plus idiosyncratic theta/beta
peaks), while the clinical task lives in the *amplitude* of the alpha
rhythm (alpha is strong with eyes closed, suppressed with eyes open).
Those two facts are physiologically real and, crucially, partly
separable -- which is exactly why anonymisation is possible at all.
====================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------
# Global recording parameters (typical of a downsampled clinical EEG)
# --------------------------------------------------------------------
FS = 128                 # sampling rate (Hz)
EPOCH_SEC = 4            # length of one analysis window (s)
N_SAMPLES = FS * EPOCH_SEC          # 512 samples per epoch
NFREQ = N_SAMPLES // 2 + 1          # rfft output length
FREQS = np.fft.rfftfreq(N_SAMPLES, 1 / FS)   # frequency axis (Hz)

# Frequency band edges (Hz) used for coarse band-power features
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# We analyse 1-40 Hz; everything above is mains/EMG territory here.
FMASK = (FREQS >= 1) & (FREQS <= 40)


# ====================================================================
# ACT 1 -- SYNTHESISE A LABELLED EEG DATASET
# ====================================================================
def _gauss(f0, sigma):
    """Unit-height Gaussian bump on the frequency grid (peak = 1)."""
    return np.exp(-0.5 * ((FREQS - f0) / sigma) ** 2)


def _realise(power_env, rng):
    """Turn a target POWER spectrum into a realistic time-domain epoch.

    We drive the amplitude spectrum with complex Gaussian noise, so the
    realised periodogram fluctuates around `power_env` the way a real
    recording does (chi-square scatter), instead of being identical
    every epoch. Mean PSD over many epochs == power_env.
    """
    amp_env = np.sqrt(power_env)
    noise = (rng.standard_normal(NFREQ) + 1j * rng.standard_normal(NFREQ)) / np.sqrt(2)
    return np.fft.irfft(amp_env * noise, n=N_SAMPLES)


def make_subject_profiles(n_subjects, rng):
    """Each subject gets a stable spectral *fingerprint*.

    Identity = WHERE the peaks sit (alpha/theta/beta centre freqs).
    Peak *powers* are kept similar across subjects on purpose, so the
    only reliable identity cue is peak *location*, not overall power.
    """
    return [
        {
            "iaf":     rng.uniform(8.5, 11.5),   # individual alpha frequency
            "theta_f": rng.uniform(4.5, 7.5),    # idiosyncratic theta peak
            "beta_f":  rng.uniform(15.0, 28.0),  # idiosyncratic beta peak
            "pink":    rng.uniform(1.8, 2.2),    # 1/f amplitude
        }
        for _ in range(n_subjects)
    ]


def make_dataset(n_subjects=20, n_epochs=60, seed=0):
    """Generate (X, subject_label, task_label).

    Task: eyes-closed (1) vs eyes-open (0). Eyes-open suppresses the
    alpha rhythm -- the classic 'alpha blocking' effect -- which is our
    clinical signal of interest (utility). Powers below are signal-to-
    noise ratios over a unit broadband floor, chosen so peaks are
    clearly visible above the floor.
    """
    rng = np.random.default_rng(seed)
    profiles = make_subject_profiles(n_subjects, rng)
    f = FREQS.copy(); f[0] = f[1]            # avoid 1/0 at DC

    X, y_subj, y_task = [], [], []
    for s, p in enumerate(profiles):
        for _ in range(n_epochs):
            eyes_closed = rng.integers(0, 2)         # random task per epoch
            # ---- clinical signal: alpha PEAK POWER depends on task ----
            alpha_pow = (30.0 if eyes_closed else 6.0) * rng.uniform(0.85, 1.15)

            power_env = (
                1.0                                   # white broadband floor
                + (p["pink"] / f ** 0.8) ** 2         # 1/f background
                + alpha_pow * _gauss(p["iaf"], 0.7)   # alpha: power = TASK, loc = identity
                + 15.0 * _gauss(p["theta_f"], 0.8)    # theta peak  (identity)
                + 9.0 * _gauss(p["beta_f"], 1.2)      # beta peak   (identity)
            )
            X.append(_realise(power_env, rng))
            y_subj.append(s)
            y_task.append(int(eyes_closed))

    return np.array(X), np.array(y_subj), np.array(y_task)


# ====================================================================
# Feature extraction -- the "released representation"
# ====================================================================
def psd(X):
    """Power spectral density per epoch (the released spectral view).

    We release a spectral representation because (a) almost all EEG
    biomarkers are spectral and (b) it lets the attacker and the task
    decoder see exactly the same thing -- a fair, honest comparison.
    """
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2 / N_SAMPLES
    return P[:, FMASK]                      # keep 1-40 Hz


def log_psd(X):
    return np.log(psd(X) + 1e-12)


def band_powers(X):
    """Coarse 4-number summary: total power in each classic band."""
    P = psd(X)
    f = FREQS[FMASK]
    out = []
    for lo, hi in BANDS.values():
        out.append(P[:, (f >= lo) & (f < hi)].sum(axis=1))
    return np.log(np.array(out).T + 1e-12)


# ====================================================================
# ACT 2 -- THE TWO EVALUATORS
# ====================================================================
def evaluate(features, y_subj, y_task, seed=0):
    """Train+test a re-ID attacker and a task decoder on the SAME
    released features. Returns (reid_accuracy, task_accuracy).

    Both models are *retrained* on the anonymised features -- this is
    the honest threat model: the attacker adapts to your defence.
    """
    idx = np.arange(len(y_subj))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                              stratify=y_subj)

    # Privacy: can the attacker name the subject? (chance = 1/n_subjects)
    reid = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    reid.fit(features[tr], y_subj[tr])
    reid_acc = reid.score(features[te], y_subj[te])

    # Utility: can the clinician's decoder still read the task? (chance = 0.5)
    task = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    task.fit(features[tr], y_task[tr])
    task_acc = task.score(features[te], y_task[te])

    return reid_acc, task_acc


# ====================================================================
# ACT 3 -- THE ANONYMISATION TRANSFORMS  f(.)
# Each returns the released FEATURE matrix for a given raw X.
# ====================================================================
def anon_raw(X):
    """No anonymisation -- baseline. High utility AND high identity."""
    return log_psd(X)


def anon_gaussian_noise(X, snr_db, rng):
    """Naive perturbation: add white noise at a target SNR (time domain)."""
    p_sig = np.mean(X ** 2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noisy = X + np.sqrt(p_noise) * rng.standard_normal(X.shape)
    return log_psd(noisy)


def anon_dp_features(X, epsilon, rng, clip=8.0):
    """Differential-privacy-style Laplace mechanism on the features.

    Clip features to a bounded range so sensitivity is finite, then add
    Laplace noise scaled by sensitivity/epsilon. Smaller epsilon = more
    privacy = more noise. (A teaching-grade DP demo, not a audited one.)
    """
    feats = np.clip(log_psd(X), -clip, clip)
    sensitivity = 2 * clip
    scale = sensitivity / epsilon
    return feats + rng.laplace(0.0, scale, feats.shape)


def anon_spectral_smooth(X, sigma_hz):
    """Blur the spectrum along frequency.

    Washes out the precise *location* of each subject's peaks (identity)
    while preserving integrated band power (the alpha task signal).
    """
    df = FREQS[1] - FREQS[0]
    return gaussian_filter1d(log_psd(X), sigma=sigma_hz / df, axis=1)


def anon_band_powers(X):
    """Collapse to 4 band powers only.

    Throws away ALL fine spectral location (kills the frequency
    fingerprint) but keeps alpha-band power, so the task survives.
    """
    return band_powers(X)


def anon_iaf_align(X):
    """Frequency-warp each epoch so its alpha peak sits at a common
    reference (10 Hz). Removes the individual-alpha-frequency biometric
    while preserving alpha magnitude. Theta/beta peaks still leak a bit
    -- a realistic 'partial' defence."""
    P = log_psd(X)
    f = FREQS[FMASK]
    alpha_mask = (f >= 8) & (f <= 13)
    ref_bin = np.argmin(np.abs(f - 10.0))
    aligned = np.empty_like(P)
    for i, row in enumerate(P):
        peak_bin = np.where(alpha_mask)[0][np.argmax(row[alpha_mask])]
        aligned[i] = np.roll(row, ref_bin - peak_bin)
    return aligned


def anon_phase_randomize(X, rng):
    """Surrogate-data trick: randomise phases, keep the magnitude
    spectrum. Looks like anonymisation but PRESERVES the power spectrum
    -- and the power spectrum is exactly the fingerprint. Included to
    show a method that *fails*."""
    spec = np.fft.rfft(X, axis=-1)
    phases = rng.uniform(0, 2 * np.pi, spec.shape)
    surrogate = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=N_SAMPLES)
    return log_psd(surrogate)


# ====================================================================
# ACT 4 -- RUN THE EXPERIMENT
# ====================================================================
def main():
    rng = np.random.default_rng(42)
    n_subjects = 20
    X, y_subj, y_task = make_dataset(n_subjects=n_subjects, n_epochs=60, seed=1)
    print(f"Dataset: {X.shape[0]} epochs x {X.shape[1]} samples, "
          f"{n_subjects} subjects.")
    chance_reid = 1.0 / n_subjects

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
    print(f"\nChance levels:  re-ID = {chance_reid:.3f}   task = 0.500")
    print(f"{'method':<26}{'re-ID (privacy)':>16}{'task (utility)':>16}")
    print("-" * 58)
    for name, (r, t) in results.items():
        print(f"{name:<26}{r:>16.3f}{t:>16.3f}")

    # ================================================================
    # FIGURE 1 -- see the data: identity vs task in the spectrum
    # ================================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    f = FREQS[FMASK]

    # left: two subjects, same task -> different peak locations = IDENTITY
    for s, color in [(0, "#534AB7"), (5, "#D85A30")]:
        m = (y_subj == s) & (y_task == 1)
        ax[0].plot(f, psd(X[m]).mean(0), color=color, lw=2, label=f"subject {s}")
    ax[0].set_title("Identity lives in peak LOCATION\n(two subjects, eyes closed)")
    ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Power")
    ax[0].legend(); ax[0].set_xlim(1, 40)

    # right: one subject, two tasks -> alpha amplitude changes = TASK
    for task, color, lab in [(1, "#0F6E56", "eyes closed"), (0, "#185FA5", "eyes open")]:
        m = (y_subj == 0) & (y_task == task)
        ax[1].plot(f, psd(X[m]).mean(0), color=color, lw=2, label=lab)
    ax[1].axvspan(8, 13, color="0.85", alpha=0.5, label="alpha band")
    ax[1].set_title("Task lives in alpha AMPLITUDE\n(one subject, two conditions)")
    ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("Power")
    ax[1].legend(); ax[1].set_xlim(1, 40)
    fig.tight_layout(); fig.savefig("fig1_signal_structure.png", dpi=130)

    # ================================================================
    # FIGURE 2 -- the privacy-utility plane (the money plot)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.axhline(0.5, ls=":", c="grey"); ax.axvline(chance_reid, ls=":", c="grey")
    ax.text(chance_reid + .01, .52, "re-ID chance", color="grey", fontsize=9)
    ax.text(.78, .505, "task chance", color="grey", fontsize=9)
    ax.add_patch(plt.Rectangle((0, 0.85), chance_reid, 0.15, color="#E1F5EE"))
    ax.text(0.005, 0.875, "ideal\ncorner", fontsize=9, color="#0F6E56")

    for (name, (r, t)), mk in zip(results.items(), ["*", "X", "P", "s", "D"]):
        ax.scatter(r, t, s=160, marker=mk, zorder=5, label=name)

    def plot_curve(curve, color, label):
        c = np.array([(r, t) for _, r, t in curve])
        ax.plot(c[:, 0], c[:, 1], "-o", color=color, alpha=.8, label=label, ms=4)

    plot_curve(noise_curve, "#BA7517", "additive noise (sweep SNR)")
    plot_curve(dp_curve, "#993556", "DP Laplace (sweep epsilon)")
    plot_curve(smooth_curve, "#185FA5", "spectral smooth (sweep sigma)")

    ax.set_xlabel("Re-identification accuracy  (lower = more private) -->")
    ax.set_ylabel("Task accuracy  (higher = more useful) -->")
    ax.set_title("The privacy-utility plane for EEG anonymisation")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig2_privacy_utility.png", dpi=130)

    # ================================================================
    # FIGURE 3 -- before/after a good transform (spectral smoothing)
    # ================================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    smoothed = anon_spectral_smooth(X, 3.0)
    for s, color in [(0, "#534AB7"), (5, "#D85A30")]:
        m = (y_subj == s) & (y_task == 1)
        ax[0].plot(f, log_psd(X[m]).mean(0), color=color, lw=2, label=f"subject {s}")
        ax[1].plot(f, smoothed[m].mean(0), color=color, lw=2, label=f"subject {s}")
    ax[0].set_title("Before: subjects separable by peaks")
    ax[1].set_title("After 3 Hz smoothing: peaks erased,\nalpha bump (task) preserved")
    for a in ax:
        a.axvspan(8, 13, color="0.85", alpha=0.5); a.set_xlabel("Frequency (Hz)")
        a.set_xlim(1, 40); a.legend()
    ax[0].set_ylabel("log power")
    fig.tight_layout(); fig.savefig("fig3_before_after.png", dpi=130)

    print("\nSaved: fig1_signal_structure.png, fig2_privacy_utility.png, "
          "fig3_before_after.png")


if __name__ == "__main__":
    main()
