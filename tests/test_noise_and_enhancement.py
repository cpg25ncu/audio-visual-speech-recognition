import numpy as np
import pytest

from avsr.audio.enhance import spectral_subtraction
from avsr.audio.features import frame_signal, power_spectrum
from avsr.audio.noise import NoiseBank, active_power, measured_snr, mix_at_snr, pink_noise

SR = 16000


def speech_like(rng, n=3 * SR):
    """A 'word' (modulated harmonic burst) surrounded by silence, like the real clips."""
    x = np.zeros(n, np.float32)
    t = np.arange(SR // 2) / SR
    burst = np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t)) * np.hanning(len(t))
    x[SR: SR + len(t)] = burst
    return x + 1e-4 * rng.standard_normal(n).astype(np.float32)


@pytest.mark.parametrize("snr", [20, 10, 0, -5])
def test_mix_hits_target_active_level_snr(snr):
    rng = np.random.default_rng(0)
    clean = speech_like(rng)
    noisy = mix_at_snr(clean, rng.standard_normal(len(clean)).astype(np.float32), snr)
    assert measured_snr(clean, noisy) == pytest.approx(snr, abs=0.05)


def test_active_level_ignores_silence():
    rng = np.random.default_rng(0)
    x = speech_like(rng)
    assert active_power(x) > 3 * np.mean(x ** 2)      # word occupies a sixth of the clip


def test_pink_noise_has_falling_spectrum():
    x = pink_noise(2 ** 16, np.random.default_rng(0))
    ps = np.abs(np.fft.rfft(x)) ** 2
    low, high = ps[10:100].mean(), ps[10000:20000].mean()
    assert low > 50 * high


def test_noise_bank_roles_use_disjoint_material():
    ambient = np.arange(1000, dtype=np.float32)
    pools = {"train": {f"a{i}": np.ones(100, np.float32) * i for i in range(8)},
             "eval": {f"b{i}": np.ones(100, np.float32) * i for i in range(8)}}
    bank = NoiseBank(ambient, pools, ambient_train_fraction=0.6, talkers=3)
    rng = np.random.default_rng(0)
    assert bank.sample("ambient", 100, rng, "train").max() < 600
    assert bank.sample("ambient", 100, rng, "eval").min() >= 600
    assert set(bank.available) == {"white", "pink", "babble", "ambient"}


def test_babble_never_contains_the_target_clip():
    pools = {"eval": {"target": np.full(100, 1e3, np.float32), **{f"o{i}": np.random.default_rng(i).standard_normal(100)
                                                                  .astype(np.float32) for i in range(3)}}}
    bank = NoiseBank(None, pools, talkers=3)
    for s in range(20):   # with the target excluded, the 3 talkers must be exactly the 3 others
        noise = bank.sample("babble", 100, np.random.default_rng(s), "eval", exclude_id="target")
        assert np.abs(noise).max() < 10


def test_spectral_subtraction_improves_snr_of_tone_in_white_noise():
    rng = np.random.default_rng(0)
    n = SR * 2
    t = np.arange(n) / SR
    tone = np.where((t > 0.5) & (t < 1.5), np.sin(2 * np.pi * 1000 * t), 0).astype(np.float32)
    noise = 0.3 * rng.standard_normal(n).astype(np.float32)
    ps_noisy = power_spectrum(frame_signal(tone + noise, 400, 160), 512)
    ps_noise = power_spectrum(frame_signal(noise, 400, 160), 512)
    enhanced = spectral_subtraction(ps_noisy)
    bin_1k = round(1000 * 512 / SR)
    off = [b for b in range(5, 250) if abs(b - bin_1k) > 3]
    before = ps_noisy[:, bin_1k].mean() / ps_noise[:, off].mean()
    after = enhanced[:, bin_1k].mean() / enhanced[:, off].mean()
    assert after > 5 * before
