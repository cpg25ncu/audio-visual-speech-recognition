import numpy as np
import pytest

from avsr.audio.features import (
    MFCCConfig,
    MFCCExtractor,
    cepstra,
    deltas,
    filter_edges,
    filterbank,
    frame_signal,
    frame_times,
    hz_to_mel,
    mel_to_hz,
    num_frames,
    power_spectrum,
    pre_emphasis,
)

SR, NFFT = 16000, 512


def test_mel_scale_roundtrip_and_reference_points():
    f = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)
    assert hz_to_mel(1000.0) == pytest.approx(1000.0, abs=0.5)   # mel scale is anchored at 1 kHz


def test_framing_shapes_and_centres():
    x = np.arange(16000, dtype=np.float32)
    frames = frame_signal(x, 400, 160)
    assert frames.shape == (num_frames(16000, 400, 160), 400)
    assert np.array_equal(frames[1, :5], x[160:165])
    t = frame_times(len(frames), 400, 160, SR)
    assert t[0] == pytest.approx(0.0125) and np.allclose(np.diff(t), 0.01)


def test_triangular_mel_filterbank_matches_librosa_htk():
    librosa = pytest.importorskip("librosa")
    ours = filterbank(SR, NFFT, 40, 0, 8000, "mel", "triangular")
    ref = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=40, fmin=0, fmax=8000, htk=True, norm=None)
    assert np.allclose(ours, ref, atol=1e-5)


@pytest.mark.parametrize("scale", ["mel", "linear"])
def test_rectangular_filterbank_partitions_the_spectrum(scale):
    fb = filterbank(SR, NFFT, 26, 0, 8000, scale, "rectangular")
    covered = fb.sum(0)
    assert set(np.unique(fb)) <= {0.0, 1.0}
    assert covered.max() == 1.0          # non-overlapping
    assert (fb.sum(1) >= 1).all()        # no empty band


def test_mel_bands_widen_with_frequency_linear_bands_do_not():
    mel_w = np.diff(filter_edges(20, 0, 8000, "mel"))
    lin_w = np.diff(filter_edges(20, 0, 8000, "linear"))
    assert (np.diff(mel_w) > 0).all()
    assert np.allclose(lin_w, lin_w[0])


def test_power_spectrum_peak_at_tone_frequency():
    t = np.arange(400) / SR
    frames = np.sin(2 * np.pi * 1000 * t)[None, :]
    ps = power_spectrum(frames, NFFT)
    assert np.argmax(ps[0]) == round(1000 * NFFT / SR)


def test_dct_compacts_smooth_envelope():
    logfb = np.log(1 + np.exp(-np.linspace(-3, 3, 40) ** 2))[None, :]
    c = cepstra(np.repeat(logfb, 2, 0), 40)
    energy = np.cumsum(c[0] ** 2) / np.sum(c[0] ** 2)
    assert energy[12] > 0.99              # 13 coefficients keep >99% of a smooth envelope


def test_deltas_of_linear_ramp_equal_slope():
    feat = np.outer(np.arange(50), [1.0, -2.0]).astype(np.float32)
    d = deltas(feat, 2)
    assert np.allclose(d[5:-5], [1.0, -2.0])


def test_pre_emphasis_attenuates_dc():
    y = pre_emphasis(np.ones(100, np.float32), 0.97)
    assert np.allclose(y[1:], 0.03)


def test_extractor_output_dimensions_and_normalisation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32)
    cfg = MFCCConfig()
    feats = MFCCExtractor(cfg)(x)
    assert feats.shape == (num_frames(SR, cfg.frame_len, cfg.hop), cfg.dim) == (99, 39)
    assert np.allclose(feats.mean(0), 0, atol=1e-4)
    assert np.allclose(feats.std(0), 1, atol=1e-2)


def test_mfcc_close_to_librosa_up_to_conventions():
    """Same framing/filterbank/DCT conventions -> cepstra agree (librosa uses power_to_db, i.e. 10log10)."""
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR).astype(np.float32)
    cfg = MFCCConfig(pre_emphasis=0.0, use_energy=False, deltas=0, cmvn="none", n_ceps=13)
    ours = MFCCExtractor(cfg)(x)
    frames = frame_signal(x, cfg.frame_len, cfg.hop)
    ps = power_spectrum(frames, NFFT)
    mel = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=40, fmin=0, fmax=8000, htk=True, norm=None)
    ref = librosa.feature.mfcc(S=np.log(np.maximum(mel @ ps.T, 1e-10)), n_mfcc=13, dct_type=2, norm="ortho").T
    assert np.allclose(ours, ref, atol=1e-3)
