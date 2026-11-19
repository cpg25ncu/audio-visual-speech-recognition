"""Decision-level (late) fusion of audio and visual posteriors.

Streams are combined log-linearly (a weighted product of experts):

    log P(w | a, v)  ~  alpha * log P_A(w | a) + (1 - alpha) * log P_V(w | v)

alpha = 1 is audio only, alpha = 0 is visual only. Ways of choosing alpha compared in this project:

* global   - one alpha for all conditions, fitted on validation data;
* oracle   - one alpha per test condition (noise type, SNR, video quality), chosen on validation
             data for that condition. It needs to know the test condition, so it is an upper bound
             for any fixed weighting scheme;
* adaptive - alpha is computed per utterance from stream reliability, measured by the entropy of
             each stream's posterior: a confused (flat) posterior shifts weight to the other stream.
             alpha_i = sigmoid(a * (H_V - H_A) + b), with (a, b) fitted on validation data. No
             knowledge of the test condition is used. Because the rule is symmetric in the two
             entropies, what is learnt from *acoustic* degradation (flat audio posteriors) also
             applies to *visual* degradation never seen during fitting.

Global and adaptive parameters are fitted by minimising validation log-loss rather than
maximising accuracy: when one stream is near-perfect, accuracy is identical for most weightings
and an accuracy search degenerates, whereas log-loss still prefers well-calibrated weights.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8
ALPHAS = np.round(np.linspace(0, 1, 21), 3)


def fuse_log_linear(pa: np.ndarray, pv: np.ndarray, alpha) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    logp = alpha * np.log(pa + EPS) + (1 - alpha) * np.log(pv + EPS)
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def normalised_entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + EPS)).sum(1) / np.log(p.shape[1])


def entropy_weights(pa: np.ndarray, pv: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * (normalised_entropy(pv) - normalised_entropy(pa)) + b)))


def _acc(p, y):
    return float((p.argmax(1) == y).mean())


def _nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + EPS).mean())


def select_alpha(pairs, ys, criterion: str = "acc") -> float:
    """Alpha optimising the validation criterion over the given (pa, pv) pairs.

    ``acc``: maximise accuracy (ties -> closest to 0.5); ``nll``: minimise log-loss.
    """
    if criterion == "nll":
        losses = [np.mean([_nll(fuse_log_linear(pa, pv, a), y) for (pa, pv), y in zip(pairs, ys)]) for a in ALPHAS]
        return float(ALPHAS[int(np.argmin(losses))])
    scores = [np.mean([_acc(fuse_log_linear(pa, pv, a), y) for (pa, pv), y in zip(pairs, ys)]) for a in ALPHAS]
    best = max(scores)
    candidates = [a for a, s in zip(ALPHAS, scores) if s >= best - 1e-12]
    return float(min(candidates, key=lambda a: abs(a - 0.5)))


def select_adaptive_params(pairs, ys, a_grid=None, b_grid=None) -> tuple[float, float]:
    """(a, b) minimising validation log-loss of the entropy-weighted fusion."""
    a_grid = np.linspace(0, 80, 41) if a_grid is None else a_grid
    b_grid = np.linspace(-4, 4, 33) if b_grid is None else b_grid
    best, best_ab = np.inf, (0.0, 0.0)
    for a in a_grid:
        for b in b_grid:
            loss = np.mean([_nll(fuse_log_linear(pa, pv, entropy_weights(pa, pv, a, b)), y)
                            for (pa, pv), y in zip(pairs, ys)])
            if loss < best - 1e-12:
                best, best_ab = loss, (float(a), float(b))
    return best_ab
