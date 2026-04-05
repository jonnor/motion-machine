"""
test_scaler.py - Test suite for scaler.py
Runs under MicroPython. Same conventions as test_pca.py.
Each test is an independent test_*() function.
"""

import math
import array
import random
import sys

sys.path.insert(0, '.')
from scaler import (
    StandardScaler,
    _alloc_f, _alloc_f2,
    _compute_mean, _compute_var, _compute_scale, _apply_scaling,
)

# ---------------------------------------------------------------------------
# Assertion helpers
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
    u1 = random.random()
    u2 = random.random()
    if u1 < 1e-12:
        u1 = 1e-12
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z

def _make_dataset(n_samples, n_features, means, sigmas, seed=0):
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        for j in range(n_features):
            val = int(_gauss(means[j], sigmas[j]))
            data.append(max(-32768, min(32767, val)))
    return array.array('h', data)

def _col(flat, n_features, col):
    n = len(flat) // n_features
    return [flat[s * n_features + col] for s in range(n)]

def _mean(vals):
    return sum(vals) / len(vals)

def _std(vals):
    m = _mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))

# ---------------------------------------------------------------------------
# Tests: _alloc_f / _alloc_f2
# ---------------------------------------------------------------------------

def test_alloc_f_length():
    buf = _alloc_f(7)
    assert_true(len(buf) == 7)

def test_alloc_f_zeros():
    buf = _alloc_f(4)
    assert_true(all(v == 0.0 for v in buf))

def test_alloc_f_is_array():
    assert_true(isinstance(_alloc_f(3), array.array))

def test_alloc_f2_length():
    buf = _alloc_f2(5, 3)
    assert_true(len(buf) == 15)

# ---------------------------------------------------------------------------
# Tests: _compute_mean
# ---------------------------------------------------------------------------

def test_compute_mean_known_values():
    # rows [10, 20], [30, 40] -> means [20, 30]
    X = array.array('h', [10, 20, 30, 40])
    m = _alloc_f(2)
    _compute_mean(X, 2, 2, m)
    assert_close(m[0], 20.0)
    assert_close(m[1], 30.0)

def test_compute_mean_single_row():
    X = array.array('h', [5, 10, 15])
    m = _alloc_f(3)
    _compute_mean(X, 1, 3, m)
    assert_close(m[0], 5.0)
    assert_close(m[1], 10.0)
    assert_close(m[2], 15.0)

def test_compute_mean_zero_mean():
    # col0: [-50, 50] -> mean=0;  col1: [0, 0] -> mean=0
    X = array.array('h', [-50, 0, 50, 0])
    m = _alloc_f(2)
    _compute_mean(X, 2, 2, m)
    assert_close(m[0], 0.0)
    assert_close(m[1], 0.0)

def test_compute_mean_negative_values():
    X = array.array('h', [-300, -100, -200, -400])
    m = _alloc_f(2)
    _compute_mean(X, 2, 2, m)
    assert_close(m[0], -250.0)
    assert_close(m[1], -250.0)

# ---------------------------------------------------------------------------
# Tests: _compute_var
# ---------------------------------------------------------------------------

def test_compute_var_known_values():
    # col0: [1, 3]  -> mean=2, var = ((1-2)^2 + (3-2)^2) / 1 = 2
    X = array.array('h', [1, 0, 3, 0])
    m = _alloc_f(2)
    _compute_mean(X, 2, 2, m)
    v = _alloc_f(2)
    _compute_var(X, 2, 2, m, v)
    assert_close(v[0], 2.0)
    assert_close(v[1], 0.0)

def test_compute_var_constant_feature():
    X = array.array('h', [5, 5, 5, 5])
    m = _alloc_f(2)
    _compute_mean(X, 2, 2, m)
    v = _alloc_f(2)
    _compute_var(X, 2, 2, m, v)
    assert_close(v[0], 0.0)
    assert_close(v[1], 0.0)

def test_compute_var_three_samples():
    # [1,4], [2,4], [3,4] -> var col0 = 1.0, var col1 = 0.0
    X = array.array('h', [1, 4, 2, 4, 3, 4])
    m = _alloc_f(2)
    _compute_mean(X, 3, 2, m)
    v = _alloc_f(2)
    _compute_var(X, 3, 2, m, v)
    assert_close(v[0], 1.0)
    assert_close(v[1], 0.0)

# ---------------------------------------------------------------------------
# Tests: _compute_scale
# ---------------------------------------------------------------------------

def test_compute_scale_with_std():
    var_ = array.array('f', [4.0, 9.0, 1.0])
    scale = _alloc_f(3)
    _compute_scale(var_, 3, scale, with_std=True)
    assert_close(scale[0], 0.5)   # 1/sqrt(4)
    assert_close(scale[1], 1/3.0) # 1/sqrt(9)
    assert_close(scale[2], 1.0)   # 1/sqrt(1)

def test_compute_scale_without_std():
    var_ = array.array('f', [100.0, 200.0])
    scale = _alloc_f(2)
    _compute_scale(var_, 2, scale, with_std=False)
    assert_close(scale[0], 1.0)
    assert_close(scale[1], 1.0)

def test_compute_scale_zero_variance_no_div_zero():
    var_ = array.array('f', [0.0, 4.0])
    scale = _alloc_f(2)
    _compute_scale(var_, 2, scale, with_std=True)
    assert_close(scale[0], 1.0)   # zero-var -> scale=1 (no crash)
    assert_close(scale[1], 0.5)

# ---------------------------------------------------------------------------
# Tests: StandardScaler.fit
# ---------------------------------------------------------------------------

def test_fit_sets_n_features():
    X = _make_dataset(20, 5, [0]*5, [100]*5)
    s = StandardScaler()
    s.fit(X, 20)
    assert_true(s.n_features_ == 5)

def test_fit_sets_n_samples():
    X = _make_dataset(20, 5, [0]*5, [100]*5)
    s = StandardScaler()
    s.fit(X, 20)
    assert_true(s.n_samples_ == 20)

def test_fit_allocates_mean():
    X = _make_dataset(20, 5, [0]*5, [100]*5)
    s = StandardScaler()
    s.fit(X, 20)
    assert_true(s.mean_ is not None and len(s.mean_) == 5)

def test_fit_allocates_var():
    X = _make_dataset(20, 5, [0]*5, [100]*5)
    s = StandardScaler()
    s.fit(X, 20)
    assert_true(s.var_ is not None and len(s.var_) == 5)

def test_fit_allocates_scale():
    X = _make_dataset(20, 5, [0]*5, [100]*5)
    s = StandardScaler()
    s.fit(X, 20)
    assert_true(s.scale_ is not None and len(s.scale_) == 5)

def test_fit_mean_accuracy():
    X = array.array('h', [10, 20, 30, 40])  # 2 samples, 2 features
    s = StandardScaler()
    s.fit(X, 2)
    assert_close(s.mean_[0], 20.0)
    assert_close(s.mean_[1], 30.0)

def test_fit_returns_self():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    ret = s.fit(X, 20)
    assert_true(ret is s)

# ---------------------------------------------------------------------------
# Tests: transform — output statistics
# ---------------------------------------------------------------------------

def test_transform_output_length():
    X = _make_dataset(30, 4, [100]*4, [50]*4)
    s = StandardScaler()
    Xs = s.fit_transform(X, 30)
    assert_true(len(Xs) == 30 * 4)

def test_transform_mean_near_zero():
    X = _make_dataset(60, 4, [200, -50, 1000, 0], [80, 30, 200, 10], seed=1)
    s = StandardScaler()
    Xs = s.fit_transform(X, 60)
    for j in range(4):
        col = _col(Xs, 4, j)
        assert_close(_mean(col), 0.0, tol=0.05,
                     msg="col " + str(j) + " mean=" + str(_mean(col)))

def test_transform_std_near_one():
    X = _make_dataset(60, 4, [200, -50, 1000, 0], [80, 30, 200, 10], seed=1)
    s = StandardScaler()
    Xs = s.fit_transform(X, 60)
    for j in range(4):
        col = _col(Xs, 4, j)
        assert_close(_std(col), 1.0, tol=0.05,
                     msg="col " + str(j) + " std=" + str(_std(col)))

def test_transform_with_mean_false_preserves_scale():
    X = _make_dataset(40, 3, [500]*3, [100]*3, seed=2)
    s = StandardScaler(with_mean=False)
    Xs = s.fit_transform(X, 40)
    # Mean should NOT be zero (raw values shifted only by scale)
    col = _col(Xs, 3, 0)
    assert_true(abs(_mean(col)) > 1.0, "expected non-zero mean with with_mean=False")

def test_transform_with_std_false_preserves_spread():
    X = _make_dataset(40, 2, [0, 0], [10, 100], seed=3)
    s = StandardScaler(with_std=False)
    Xs = s.fit_transform(X, 40)
    # Std should NOT be 1 — original variance preserved
    col0_std = _std(_col(Xs, 2, 0))
    col1_std = _std(_col(Xs, 2, 1))
    assert_true(abs(col0_std - col1_std) > 5.0,
                "expected different stds with with_std=False")

def test_transform_uses_preallocated_buffer():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    s.fit(X, 20)
    out = _alloc_f2(20, 3)
    ret = s.transform(X, 20, out=out)
    assert_true(ret is out)

def test_transform_preallocated_values_match():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    s.fit(X, 20)
    ref = s.transform(X, 20)
    out = _alloc_f2(20, 3)
    s.transform(X, 20, out=out)
    assert_close_arr(ref, out, tol=1e-5)

def test_fit_transform_equals_fit_then_transform():
    X = _make_dataset(30, 4, [100]*4, [40]*4, seed=5)
    sa = StandardScaler()
    Xa = sa.fit_transform(X, 30)
    sb = StandardScaler()
    sb.fit(X, 30)
    Xb = sb.transform(X, 30)
    assert_close_arr(Xa, Xb, tol=1e-5)

# ---------------------------------------------------------------------------
# Tests: transform_one
# ---------------------------------------------------------------------------

def test_transform_one_length():
    X = _make_dataset(20, 4, [0]*4, [100]*4)
    s = StandardScaler()
    s.fit(X, 20)
    sample = array.array('h', X[:4])
    out = s.transform_one(sample)
    assert_true(len(out) == 4)

def test_transform_one_matches_batch():
    X = _make_dataset(30, 4, [100, 200, 50, 300], [80, 40, 20, 150], seed=7)
    s = StandardScaler()
    Xs = s.fit_transform(X, 30)
    sample = array.array('h', X[:4])
    out = s.transform_one(sample)
    for j in range(4):
        assert_close(out[j], Xs[j], tol=1e-4,
                     msg="feature " + str(j))

def test_transform_one_uses_preallocated_buffer():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    s.fit(X, 20)
    sample = array.array('h', X[:3])
    out = _alloc_f(3)
    ret = s.transform_one(sample, out=out)
    assert_true(ret is out)

# ---------------------------------------------------------------------------
# Tests: inverse_transform
# ---------------------------------------------------------------------------

def test_inverse_transform_round_trip():
    X = _make_dataset(40, 5, [100, -200, 500, 0, 800], [30, 50, 100, 10, 200], seed=8)
    s = StandardScaler()
    Xs = s.fit_transform(X, 40)
    Xback = s.inverse_transform(Xs, 40)
    for i in range(40 * 5):
        assert_close(Xback[i], float(X[i]), tol=0.5,
                     msg="index " + str(i))

def test_inverse_transform_output_length():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    Xs = s.fit_transform(X, 20)
    Xback = s.inverse_transform(Xs, 20)
    assert_true(len(Xback) == 20 * 3)

def test_inverse_transform_uses_preallocated_buffer():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    Xs = s.fit_transform(X, 20)
    out = _alloc_f2(20, 3)
    ret = s.inverse_transform(Xs, 20, out=out)
    assert_true(ret is out)

def test_inverse_transform_one_round_trip():
    X = _make_dataset(30, 4, [50, 200, -100, 0], [20, 80, 40, 5], seed=9)
    s = StandardScaler()
    s.fit(X, 30)
    sample = array.array('h', X[:4])
    scaled = s.transform_one(sample)
    back   = s.inverse_transform_one(scaled)
    for j in range(4):
        assert_close(back[j], float(sample[j]), tol=0.5,
                     msg="feature " + str(j))

def test_inverse_transform_one_uses_preallocated_buffer():
    X = _make_dataset(20, 3, [0]*3, [50]*3)
    s = StandardScaler()
    s.fit(X, 20)
    sample = array.array('h', X[:3])
    scaled = s.transform_one(sample)
    out = _alloc_f(3)
    ret = s.inverse_transform_one(scaled, out=out)
    assert_true(ret is out)

# ---------------------------------------------------------------------------
# Tests: with_mean=False, with_std=False combinations
# ---------------------------------------------------------------------------

def test_with_mean_false_no_centering():
    # with_mean=False: scaled values should have same mean as input / scale
    X = array.array('h', [200, 200, 200, 200])  # 2 samples, 2 features
    s = StandardScaler(with_mean=False)
    Xs = s.fit_transform(X, 2)
    # mean of col0 should be 200 * scale[0], not 0
    assert_true(abs(Xs[0]) > 0.0)

def test_with_std_false_scale_is_one():
    X = _make_dataset(20, 3, [0]*3, [100]*3, seed=4)
    s = StandardScaler(with_std=False)
    s.fit(X, 20)
    for j in range(3):
        assert_close(s.scale_[j], 1.0)

def test_no_mean_no_std_is_passthrough():
    # with_mean=False, with_std=False -> output == float(input)
    X = array.array('h', [10, 20, 30, 40])
    s = StandardScaler(with_mean=False, with_std=False)
    Xs = s.fit_transform(X, 2)
    assert_close(Xs[0], 10.0)
    assert_close(Xs[1], 20.0)
    assert_close(Xs[2], 30.0)
    assert_close(Xs[3], 40.0)

# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

def test_edge_constant_feature_no_crash():
    # Zero-variance column — must not divide by zero
    X = array.array('h', [5, 100, 5, 200, 5, 300])  # 3 samples, 2 features
    s = StandardScaler()
    s.fit(X, 3)
    assert_true(True)

def test_edge_constant_feature_scale_is_one():
    X = array.array('h', [5, 100, 5, 200, 5, 300])
    s = StandardScaler()
    s.fit(X, 3)
    assert_close(s.scale_[0], 1.0)  # constant col -> scale=1

def test_edge_single_sample_fit():
    X = array.array('h', [10, 20, 30])
    s = StandardScaler(with_std=False)  # var undefined for n=1; skip std
    s.fit(X, 1)
    assert_true(s.n_features_ == 3)

def test_edge_large_values_no_overflow():
    X = array.array('h', [32000, -32000, 32000, -32000])
    s = StandardScaler()
    Xs = s.fit_transform(X, 2)
    assert_true(all(math.isfinite(v) for v in Xs))

def test_edge_many_features():
    n_feat = 20
    X = _make_dataset(50, n_feat, [i*10 for i in range(n_feat)],
                      [10]*n_feat, seed=12)
    s = StandardScaler()
    Xs = s.fit_transform(X, 50)
    assert_true(len(Xs) == 50 * n_feat)

# ---------------------------------------------------------------------------
# Test runner (identical pattern to test_pca.py)
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

    print("test_scaler.py  (" + str(len(tests)) + " tests)")
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
