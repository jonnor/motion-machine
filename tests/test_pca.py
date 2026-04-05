"""
test_pca.py - Test suite for pca.py
Runs under MicroPython (no unittest module assumed).
Uses only: math, array, random — all available in MicroPython.

Each test is an independent function named test_*().
The runner collects and calls them, reporting pass/fail per function.
A test fails if it raises an AssertionError (or any exception).
"""

import math
import array
import random
import sys

sys.path.insert(0, '.')
from pca import (
    PCA,
    _alloc_f, _alloc_f2,
    _dot_ff, _norm_f, _normalize_inplace,
    _mat_vec, _covariance_matrix,
    _compute_mean, _center,
)

# ---------------------------------------------------------------------------
# Minimal assertion helpers — raise AssertionError on failure
# ---------------------------------------------------------------------------

def assert_true(condition, msg="expected True"):
    if not condition:
        raise AssertionError(msg)

def assert_close(a, b, tol=1e-3, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(
            msg or ("got " + str(a) + " expected ~" + str(b) + " tol=" + str(tol))
        )

def assert_close_arr(a, b, tol=1e-3, msg=""):
    if len(a) != len(b):
        raise AssertionError("length mismatch " + str(len(a)) + " vs " + str(len(b)))
    for i in range(len(a)):
        if abs(a[i] - b[i]) > tol:
            raise AssertionError(
                msg or ("index " + str(i) + ": " + str(a[i]) + " vs " + str(b[i]))
            )

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gauss(mu, sigma):
    """Box-Muller normal sample — no random.gauss in MicroPython."""
    u1 = random.random()
    u2 = random.random()
    if u1 < 1e-12:
        u1 = 1e-12
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z

def _make_dataset(n_samples, n_features, means, sigma, seed=0):
    """Return a flat array.array('h') dataset."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        for j in range(n_features):
            val = int(_gauss(means[j], sigma))
            data.append(max(-32768, min(32767, val)))
    return array.array('h', data)

# ---------------------------------------------------------------------------
# Tests: low-level helpers
# ---------------------------------------------------------------------------

def test_alloc_f_length():
    buf = _alloc_f(5)
    assert_true(len(buf) == 5)

def test_alloc_f_zeros():
    buf = _alloc_f(8)
    assert_true(all(v == 0.0 for v in buf))

def test_alloc_f_is_array():
    buf = _alloc_f(3)
    assert_true(isinstance(buf, array.array))

def test_alloc_f2_length():
    buf = _alloc_f2(3, 4)
    assert_true(len(buf) == 12)

def test_dot_ff_basic():
    a = array.array('f', [1.0, 2.0, 3.0])
    b = array.array('f', [4.0, 5.0, 6.0])
    assert_close(_dot_ff(a, b, 3), 32.0)

def test_dot_ff_zeros():
    a = array.array('f', [1.0, 2.0, 3.0])
    b = array.array('f', [0.0, 0.0, 0.0])
    assert_close(_dot_ff(a, b, 3), 0.0)

def test_norm_f_pythagorean():
    v = array.array('f', [3.0, 4.0])
    assert_close(_norm_f(v, 2), 5.0)

def test_norm_f_unit():
    v = array.array('f', [1.0, 0.0, 0.0])
    assert_close(_norm_f(v, 3), 1.0)

def test_normalize_inplace_unit_length():
    v = array.array('f', [3.0, 4.0])
    _normalize_inplace(v, 2)
    assert_close(_norm_f(v, 2), 1.0)

def test_normalize_inplace_direction():
    v = array.array('f', [0.0, 5.0])
    _normalize_inplace(v, 2)
    assert_close(v[0], 0.0)
    assert_close(v[1], 1.0)

def test_mat_vec_identity():
    I = array.array('f', [1, 0, 0,  0, 1, 0,  0, 0, 1])
    x = array.array('f', [2.0, 5.0, -1.0])
    out = _alloc_f(3)
    _mat_vec(I, x, out, 3, 3)
    assert_close_arr(out, x)

def test_mat_vec_scale():
    M = array.array('f', [2, 0,  0, 2])
    x = array.array('f', [3.0, -1.0])
    out = _alloc_f(2)
    _mat_vec(M, x, out, 2, 2)
    assert_close(out[0], 6.0)
    assert_close(out[1], -2.0)

# ---------------------------------------------------------------------------
# Tests: _compute_mean
# ---------------------------------------------------------------------------

def test_compute_mean_col0():
    X = array.array('h', [2, 4, 4, 6, 6, 8])
    m = _alloc_f(2)
    _compute_mean(X, 3, 2, m)
    assert_close(m[0], 4.0)

def test_compute_mean_col1():
    X = array.array('h', [2, 4, 4, 6, 6, 8])
    m = _alloc_f(2)
    _compute_mean(X, 3, 2, m)
    assert_close(m[1], 6.0)

def test_compute_mean_single_sample():
    X = array.array('h', [10, 20, 30])
    m = _alloc_f(3)
    _compute_mean(X, 1, 3, m)
    assert_close(m[0], 10.0)
    assert_close(m[1], 20.0)
    assert_close(m[2], 30.0)

# ---------------------------------------------------------------------------
# Tests: _center
# ---------------------------------------------------------------------------

def test_center_values():
    X = array.array('h', [2, 4, 4, 6, 6, 8])
    m = _alloc_f(2)
    _compute_mean(X, 3, 2, m)
    c = _alloc_f2(3, 2)
    _center(X, 3, 2, m, c)
    assert_close(c[0], -2.0)
    assert_close(c[1], -2.0)
    assert_close(c[2],  0.0)
    assert_close(c[3],  0.0)
    assert_close(c[4],  2.0)
    assert_close(c[5],  2.0)

def test_center_mean_is_zero():
    X = _make_dataset(20, 4, [100, 200, 50, 300], 80, seed=3)
    m = _alloc_f(4)
    _compute_mean(X, 20, 4, m)
    c = _alloc_f2(20, 4)
    _center(X, 20, 4, m, c)
    for j in range(4):
        col_mean = sum(c[s*4+j] for s in range(20)) / 20
        assert_close(col_mean, 0.0, tol=1e-2)

# ---------------------------------------------------------------------------
# Tests: _covariance_matrix
# ---------------------------------------------------------------------------

def test_cov_perfectly_correlated():
    X = array.array('f', [-1.0,-1.0, 0.0,0.0, 1.0,1.0])
    cov = _alloc_f2(2, 2)
    _covariance_matrix(X, 3, 2, cov)
    assert_close(cov[0], 1.0)
    assert_close(cov[1], 1.0)
    assert_close(cov[2], 1.0)
    assert_close(cov[3], 1.0)

def test_cov_anticorrelated():
    X = array.array('f', [-1.0,1.0, 0.0,0.0, 1.0,-1.0])
    cov = _alloc_f2(2, 2)
    _covariance_matrix(X, 3, 2, cov)
    assert_close(cov[1], -1.0)

def test_cov_symmetric():
    X = _make_dataset(30, 4, [0,0,0,0], 200, seed=7)
    Xf = array.array('f', [float(v) for v in X])
    m = _alloc_f(4)
    _compute_mean(Xf, 30, 4, m)
    Xc = _alloc_f2(30, 4)
    _center(Xf, 30, 4, m, Xc)
    cov = _alloc_f2(4, 4)
    _covariance_matrix(Xc, 30, 4, cov)
    for i in range(4):
        for j in range(4):
            assert_close(cov[i*4+j], cov[j*4+i], tol=1e-3)

# ---------------------------------------------------------------------------
# Tests: PCA.fit
# ---------------------------------------------------------------------------

def test_fit_sets_n_features():
    X = _make_dataset(20, 6, [0]*6, 100, seed=0)
    pca = PCA(n_components=2)
    pca.fit(X, 20)
    assert_true(pca.n_features_ == 6)

def test_fit_sets_mean_length():
    X = _make_dataset(20, 6, [0]*6, 100, seed=0)
    pca = PCA(n_components=2)
    pca.fit(X, 20)
    assert_true(pca.mean_ is not None and len(pca.mean_) == 6)

def test_fit_sets_components_length():
    X = _make_dataset(20, 6, [0]*6, 100, seed=0)
    pca = PCA(n_components=2)
    pca.fit(X, 20)
    assert_true(len(pca.components_) == 2 * 6)

def test_fit_sets_explained_ratio_length():
    X = _make_dataset(20, 6, [0]*6, 100, seed=0)
    pca = PCA(n_components=3)
    pca.fit(X, 20)
    assert_true(len(pca.explained_ratio_) == 3)

def test_fit_single_axis_variance():
    X = array.array('h', [s*100 if j == 0 else 0
                           for s in range(10) for j in range(4)])
    pca = PCA(n_components=2, n_iter=200)
    pca.fit(X, 10)
    assert_true(pca.explained_ratio_[0] > 0.99,
                "PC1 ratio=" + str(pca.explained_ratio_[0]))

def test_fit_explained_ratio_sum():
    X = _make_dataset(40, 5, [0]*5, 300, seed=2)
    pca = PCA(n_components=3)
    pca.fit(X, 40)
    total = sum(pca.explained_ratio_[c] for c in range(3))
    assert_true(total <= 1.001, "sum=" + str(total))

def test_fit_mean_accuracy():
    X = array.array('h', [s*100 if j == 0 else 0
                           for s in range(10) for j in range(4)])
    pca = PCA(n_components=2, n_iter=200)
    pca.fit(X, 10)
    assert_close(pca.mean_[0], 450.0, tol=1.0)
    assert_close(pca.mean_[1], 0.0,   tol=0.1)

def test_fit_returns_self():
    X = _make_dataset(20, 4, [0]*4, 100, seed=0)
    pca = PCA(n_components=2)
    ret = pca.fit(X, 20)
    assert_true(ret is pca)

# ---------------------------------------------------------------------------
# Tests: PCA.transform
# ---------------------------------------------------------------------------

def test_transform_output_length():
    X = _make_dataset(30, 4, [100, 200, 50, 300], 80, seed=7)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    proj = pca.transform(X, 30)
    assert_true(len(proj) == 30 * 2)

def test_fit_transform_equals_fit_then_transform():
    X = _make_dataset(30, 4, [100, 200, 50, 300], 80, seed=7)
    pca_a = PCA(n_components=2)
    proj_a = pca_a.fit_transform(X, 30)
    pca_b = PCA(n_components=2)
    pca_b.fit(X, 30)
    proj_b = pca_b.transform(X, 30)
    assert_close_arr(proj_a, proj_b, tol=1e-4)

def test_transform_uses_preallocated_buffer():
    X = _make_dataset(30, 4, [0]*4, 100, seed=1)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    out = _alloc_f(30 * 2)
    ret = pca.transform(X, 30, out=out)
    assert_true(ret is out)

def test_transform_preallocated_values_match():
    X = _make_dataset(30, 4, [0]*4, 100, seed=1)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    ref = pca.transform(X, 30)
    out = _alloc_f(30 * 2)
    pca.transform(X, 30, out=out)
    assert_close_arr(ref, out, tol=1e-4)

# ---------------------------------------------------------------------------
# Tests: transform_one
# ---------------------------------------------------------------------------

def test_transform_one_length():
    X = _make_dataset(30, 4, [0]*4, 100, seed=2)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    sample = array.array('h', X[:4])
    out = pca.transform_one(sample)
    assert_true(len(out) == 2)

def test_transform_one_matches_batch_pc1():
    X = _make_dataset(30, 4, [100, 200, 50, 300], 80, seed=7)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    proj = pca.transform(X, 30)
    sample = array.array('h', X[:4])
    out = pca.transform_one(sample)
    assert_close(out[0], proj[0], tol=1e-3)

def test_transform_one_matches_batch_pc2():
    X = _make_dataset(30, 4, [100, 200, 50, 300], 80, seed=7)
    pca = PCA(n_components=2)
    pca.fit(X, 30)
    proj = pca.transform(X, 30)
    sample = array.array('h', X[:4])
    out = pca.transform_one(sample)
    assert_close(out[1], proj[1], tol=1e-3)

def test_transform_one_uses_preallocated_buffer():
    X = _make_dataset(20, 4, [0]*4, 100, seed=3)
    pca = PCA(n_components=2)
    pca.fit(X, 20)
    sample = array.array('h', X[:4])
    out = _alloc_f(2)
    ret = pca.transform_one(sample, out=out)
    assert_true(ret is out)

# ---------------------------------------------------------------------------
# Tests: component orthogonality and unit norm
# ---------------------------------------------------------------------------

def test_components_are_orthogonal():
    X = _make_dataset(80, 6, [0]*6, 500, seed=42)
    pca = PCA(n_components=3, n_iter=200)
    pca.fit(X, 80)
    nf = pca.n_features_
    for i in range(3):
        for j in range(i+1, 3):
            ci = [pca.components_[i*nf+k] for k in range(nf)]
            cj = [pca.components_[j*nf+k] for k in range(nf)]
            dot = sum(ci[k]*cj[k] for k in range(nf))
            assert_close(dot, 0.0, tol=0.05,
                         msg="c" + str(i) + ".c" + str(j) + "=" + str(dot))

def test_components_are_unit_norm():
    X = _make_dataset(80, 6, [0]*6, 500, seed=42)
    pca = PCA(n_components=3, n_iter=200)
    pca.fit(X, 80)
    nf = pca.n_features_
    for i in range(3):
        ci = [pca.components_[i*nf+k] for k in range(nf)]
        mag = math.sqrt(sum(v*v for v in ci))
        assert_close(mag, 1.0, tol=1e-3, msg="c" + str(i) + " norm=" + str(mag))

# ---------------------------------------------------------------------------
# Tests: cluster separability
# ---------------------------------------------------------------------------

def test_clusters_separated_on_pc1():
    random.seed(1)
    half = 20
    X_cl = array.array('h', [])
    for s in range(half):
        mu = 1000 if s < half//2 else -1000
        row = [max(-32768, min(32767, int(_gauss(mu, 100))))] + \
              [max(-32768, min(32767, int(_gauss(0, 100)))) for _ in range(3)]
        for v in row:
            X_cl.append(v)
    pca = PCA(n_components=2, n_iter=200)
    proj = pca.fit_transform(X_cl, half)
    mean_a = sum(proj[s*2] for s in range(half//2)) / (half//2)
    mean_b = sum(proj[s*2] for s in range(half//2, half)) / (half//2)
    assert_true(abs(mean_a - mean_b) > 500,
                "mean_a=" + str(mean_a) + " mean_b=" + str(mean_b))

# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

def test_edge_single_component_ratio_length():
    X = _make_dataset(20, 4, [0]*4, 200, seed=5)
    pca = PCA(n_components=1)
    pca.fit(X, 20)
    assert_true(len(pca.explained_ratio_) == 1)

def test_edge_single_component_ratio_valid():
    X = _make_dataset(20, 4, [0]*4, 200, seed=5)
    pca = PCA(n_components=1)
    pca.fit(X, 20)
    assert_true(0.0 < pca.explained_ratio_[0] <= 1.001)

def test_edge_square_matrix():
    X = _make_dataset(4, 4, [10, 20, 30, 40], 50, seed=9)
    pca = PCA(n_components=2)
    pca.fit(X, 4)
    assert_true(pca.n_features_ == 4)

def test_edge_constant_feature_no_crash():
    X = array.array('h', [1, 0, 2, 0, 3, 0, 4, 0, 5, 0])
    pca = PCA(n_components=1)
    pca.fit(X, 5)
    assert_true(True)

def test_edge_many_components():
    X = _make_dataset(50, 8, [0]*8, 300, seed=11)
    pca = PCA(n_components=5, n_iter=150)
    proj = pca.fit_transform(X, 50)
    assert_true(len(proj) == 50 * 5)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _collect_tests():
    g = globals()
    tests = [(name, g[name]) for name in g if name.startswith('test_')]
    tests.sort(key=lambda t: t[0])
    return tests

def run_all():
    tests = _collect_tests()
    passed = 0
    failed = 0
    failures = []

    print("test_pca.py  (" + str(len(tests)) + " tests)")
    print("-" * 52)

    for name, fn in tests:
        try:
            fn()
            print("  PASS  " + name)
            passed += 1
        except AssertionError as e:
            msg = str(e) if str(e) else "AssertionError"
            print("  FAIL  " + name + "  -> " + msg)
            failures.append((name, msg))
            failed += 1
        except Exception as e:
            msg = type(e).__name__ + ": " + str(e)
            print("  ERROR " + name + "  -> " + msg)
            failures.append((name, msg))
            failed += 1

    print("-" * 52)
    print("Results: " + str(passed) + " passed, " + str(failed) + " failed")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print("  " + name + ": " + msg)

    return failed == 0

if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
