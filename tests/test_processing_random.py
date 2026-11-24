"""
Tests for randomized processing: random_indices, thinning, dark counts and hot pixels.

Randomized functions are tested with fixed seeds (so every run is deterministic) on two
kinds of property:

- invariants that hold for every draw (indices in bounds, thinning never creates
  photons, copy vs. in-place, edge cases p = 0 and p = 1);
- statistical properties (rates, uniformity, dispersion), checked against theoretical
  bounds wide enough (N_SIGMA standard errors, or p-value > P_MIN) that a correct
  implementation passes for essentially any seed, rather than being tuned to SEED.
"""
import numpy as np
import pytest
from scipy import stats

from spadlib.processing import (
    add_darkcounts_fixed_frames,
    add_darkcounts_lognormal_frames,
    add_hotpixels_frames,
    random_indices,
    thin_frames_uniform,
)

SEED = 0
N_SIGMA = 5  # two-sided false-failure probability ~6e-7 per check
P_MIN = 1e-4  # minimum p-value for goodness-of-fit tests


def assert_binomial(k, n, p):
    """Assert a count `k` of `n` Bernoulli(p) trials is within N_SIGMA of n * p."""
    assert abs(k - n * p) <= N_SIGMA * np.sqrt(n * p * (1 - p)), (k, n * p)


def dispersion_z(counts):
    """
    z-score of the index-of-dispersion test: under i.i.d. Poisson-like counts,
    sum (c - mean)^2 / mean ~ chi2(N - 1), approximated as normal for large N.
    """
    n = counts.size
    d = ((counts - counts.mean()) ** 2).sum() / counts.mean()
    return (d - (n - 1)) / np.sqrt(2 * (n - 1))


# --- random_indices ---------------------------------------------------------------------

def test_random_indices_valid():
    shape = (50, 40, 30)
    idx = random_indices(shape, 0.05, seed=SEED)
    assert len(idx) == 3
    for i, s in zip(idx, shape):
        assert i.min() >= 0 and i.max() < s
    flat = np.ravel_multi_index(idx, shape)
    assert np.all(np.diff(flat) > 0)  # sorted and unique
    assert_binomial(flat.size, np.prod(shape), 0.05)


def test_random_indices_uniform_over_positions():
    # each position should be selected equally often: chi-square test of per-position counts
    n_draws, n_pos, p = 5000, 200, 0.1
    rows, cols = random_indices((n_draws, n_pos), p, seed=SEED)
    per_pos = np.bincount(cols, minlength=n_pos)
    assert stats.chisquare(per_pos).pvalue > P_MIN
    per_draw = np.bincount(rows, minlength=n_draws)
    assert stats.chisquare(per_draw).pvalue > P_MIN


def test_random_indices_shape_from_array_or_int():
    assert len(random_indices(np.zeros((3, 4)), 0.5, seed=SEED)) == 2
    assert len(random_indices(7, 0.5, seed=SEED)) == 1


def test_random_indices_p_zero_and_one():
    shape = (4, 5, 6)
    assert all(i.size == 0 for i in random_indices(shape, 0.0, seed=SEED))
    for a, b in zip(random_indices(shape, 1.0, seed=SEED), np.nonzero(np.ones(shape))):
        np.testing.assert_array_equal(a, b)


def test_random_indices_reproducible():
    a = random_indices((100, 100), 0.1, seed=SEED)
    b = random_indices((100, 100), 0.1, seed=np.random.default_rng(SEED))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    c = random_indices((100, 100), 0.1, seed=SEED + 1)
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_random_indices_invalid_p_raises(p):
    with pytest.raises(ValueError):
        random_indices((10,), p)


# --- thin_frames_uniform ----------------------------------------------------------------

@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])  # sparse-keep, dense, sparse-drop branches
@pytest.mark.parametrize("t_chunk", [3, 1000])
def test_thin_frames_uniform_keeps_fraction_p(frames, p, t_chunk):
    original = frames.copy()
    out = thin_frames_uniform(frames, p, t_chunk=t_chunk, seed=SEED)
    np.testing.assert_array_equal(frames, original)  # copy=True leaves input untouched
    assert out.shape == frames.shape and out.dtype == frames.dtype
    assert np.all(out <= frames)  # never creates photons
    assert_binomial(int(out.sum()), int(frames.sum()), p)


@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_thin_frames_uniform_in_place(frames, p):
    original = frames.copy()
    out = thin_frames_uniform(frames, p, copy=False, seed=SEED)
    assert out is frames
    assert np.all(frames <= original) and frames.sum() < original.sum()


def test_thin_frames_uniform_p_zero_and_one(frames):
    assert thin_frames_uniform(frames, 0.0, seed=SEED).sum() == 0
    np.testing.assert_array_equal(thin_frames_uniform(frames, 1.0, seed=SEED), frames)


def test_thin_frames_uniform_dark_counts():
    zeros = np.zeros((10, 256, 256), dtype=np.uint8)
    out = thin_frames_uniform(zeros, 0.5, dcr_prob=1e-3, seed=SEED)
    assert_binomial(int(out.sum()), zeros.size, 1e-3)


def test_thin_frames_uniform_bool_frames(frames):
    out = thin_frames_uniform(frames.astype(bool), 0.5, seed=SEED)
    assert out.dtype == bool
    assert_binomial(int(out.sum()), int(frames.sum()), 0.5)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_thin_frames_uniform_invalid_p_raises(frames, p):
    with pytest.raises(ValueError):
        thin_frames_uniform(frames, p)


# --- dark counts ------------------------------------------------------------------------

def test_add_darkcounts_fixed_rate_and_dispersion():
    zeros = np.zeros((2000, 64, 64), dtype=np.uint8)
    out = add_darkcounts_fixed_frames(zeros, 0.01, seed=SEED)
    assert zeros.sum() == 0  # copy=True leaves input untouched
    assert_binomial(int(out.sum()), zeros.size, 0.01)
    # same probability for every pixel: per-pixel counts are binomial, not overdispersed
    # (binomial variance is (1 - p) times the mean, so the expected z offset is negligible)
    assert abs(dispersion_z(out.sum(axis=0))) < N_SIGMA


def test_add_darkcounts_fixed_never_removes_photons(frames):
    original = frames.copy()
    out = add_darkcounts_fixed_frames(frames, 0.01, copy=False, seed=SEED)
    assert out is frames
    assert np.all(frames >= original) and frames.sum() > original.sum()


def test_add_darkcounts_lognormal_mean():
    T, H, W, mean, sigma = 2000, 64, 64, 0.01, 0.36
    out = add_darkcounts_lognormal_frames(np.zeros((T, H, W), dtype=np.uint8), mean, sigma, seed=SEED)
    # each pixel's count has variance T*mean + T^2 Var[p] (counting noise + pixel spread)
    var_p = mean ** 2 * (np.exp(sigma ** 2) - 1)
    sd_total = np.sqrt(H * W * (T * mean + T ** 2 * var_p))
    assert abs(out.sum() - T * H * W * mean) < N_SIGMA * sd_total


def test_add_darkcounts_lognormal_recovers_sigma():
    # method of moments: Var[k] = E[k] + T^2 Var[p], with Var[p] = E[p]^2 (exp(sigma^2) - 1)
    T, mean, sigma = 2000, 0.01, 0.36
    counts = add_darkcounts_lognormal_frames(
        np.zeros((T, 128, 128), dtype=np.uint8), mean, sigma, seed=SEED).sum(axis=0)
    m, v = counts.mean(), counts.var()
    sigma_hat = np.sqrt(np.log(1 + (v - m) / m ** 2))
    # at this size sigma_hat has sd ~0.003 and bias ~-0.0015 (measured over 30 seeds)
    assert abs(sigma_hat - sigma) < 0.02


def test_add_darkcounts_lognormal_sigma_zero_is_fixed():
    out = add_darkcounts_lognormal_frames(np.zeros((2000, 64, 64), dtype=np.uint8), 0.01, 0.0, seed=SEED)
    assert abs(dispersion_z(out.sum(axis=0))) < N_SIGMA


def test_add_darkcounts_lognormal_never_removes_photons(frames):
    original = frames.copy()
    out = add_darkcounts_lognormal_frames(frames, 0.01, 0.36, copy=False, seed=SEED)
    assert out is frames
    assert np.all(frames >= original) and frames.sum() > original.sum()


# --- hot pixels -------------------------------------------------------------------------

@pytest.mark.parametrize("t_chunk", [300, 1000])
def test_add_hotpixels_fraction_and_probabilities(t_chunk):
    T, H, W, hot_prob, min_prob, max_prob = 1000, 128, 128, 0.02, 0.2, 0.8
    ones = np.ones((T, H, W), dtype=np.uint8)
    out = add_hotpixels_frames(ones, hot_prob, min_prob, max_prob, t_chunk=t_chunk, seed=SEED)
    assert ones.min() == 1  # copy=True leaves input untouched
    # hot pixels' time series are replaced (and fire with p <= 0.8, so certainly contain 0s);
    # every other pixel is untouched
    hot = out.min(axis=0) == 0
    assert_binomial(int(hot.sum()), H * W, hot_prob)
    np.testing.assert_array_equal(out[:, ~hot], 1)
    # measured firing probabilities ~ Uniform(min_prob, max_prob), up to binomial noise
    # (sd <= 0.016 at T = 1000, small next to the KS test's resolution at ~300 pixels)
    p_hat = out[:, hot].mean(axis=0)
    assert stats.kstest(p_hat, stats.uniform(min_prob, max_prob - min_prob).cdf).pvalue > P_MIN


def test_add_hotpixels_in_place():
    frames = np.zeros((500, 64, 64), dtype=np.uint8)
    out = add_hotpixels_frames(frames, 0.05, 0.2, copy=False, seed=SEED)
    assert out is frames and frames.sum() > 0
