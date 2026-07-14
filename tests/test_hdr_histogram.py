"""
Pytest-style tests for hdr_histogram.py, but with zero dependencies so
they run under MicroPython as well as CPython. Just plain functions named
test_* using assert statements, discovered and run by run_all_tests().

Usage:
    python3 test_hdr_histogram.py
    micropython test_hdr_histogram.py
"""

import math
import random

from hdr_histogram import (
    HdrHistogram,
    HdrRecordError,
    HdrEmptyHistogramError,
)


def _brute_percentile(sorted_vals, p):
    idx = math.ceil((p / 100.0) * len(sorted_vals)) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def _approx_gauss(mean, stddev):
    # avoids random.gauss, which MicroPython's random module lacks
    s = 0.0
    for _ in range(12):
        s += random.random()
    return mean + (s - 6.0) * stddev


def _make_sample_dataset(n=20000, seed=None):
    if seed is not None:
        random.seed(seed)
    vals = []
    for _ in range(n):
        v = int(abs(_approx_gauss(500, 100)))
        v = max(0, min(v, 3_600_000))
        vals.append(v)
    return vals


# ---------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------

def test_construct_valid():
    h = HdrHistogram(1000)
    assert h.count() == 0


def test_construct_rejects_max_value_zero():
    try:
        HdrHistogram(0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_construct_rejects_max_value_negative():
    try:
        HdrHistogram(-5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_construct_rejects_max_value_float():
    try:
        HdrHistogram(1000.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_construct_rejects_bad_bits():
    try:
        HdrHistogram(1000, bits=8)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_construct_accepts_16_and_32_bits():
    h16 = HdrHistogram(1000, bits=16)
    h32 = HdrHistogram(1000, bits=32)
    assert h16.itemsize == 2
    assert h32.itemsize == 4


def test_record_accepts_value_zero_regardless_of_range():
    # 0 is always a valid record()-able value; there is no configurable
    # lower bound on the histogram's range
    h = HdrHistogram(1000)
    h.record(0)
    assert h.count() == 1
    assert h.percentile(100) == 0


# ---------------------------------------------------------------------
# record() validation
# ---------------------------------------------------------------------

def test_record_rejects_float():
    h = HdrHistogram(1000)
    try:
        h.record(1.5)
        assert False, "expected HdrRecordError"
    except HdrRecordError:
        pass


def test_record_rejects_bool():
    h = HdrHistogram(1000)
    try:
        h.record(True)
        assert False, "expected HdrRecordError"
    except HdrRecordError:
        pass


def test_record_rejects_negative():
    h = HdrHistogram(1000)
    try:
        h.record(-1)
        assert False, "expected HdrRecordError"
    except HdrRecordError:
        pass


def test_record_rejects_above_max():
    h = HdrHistogram(1000)
    try:
        h.record(1001)
        assert False, "expected HdrRecordError"
    except HdrRecordError:
        pass


def test_record_accepts_boundary_values():
    h = HdrHistogram(1000)
    h.record(0)
    h.record(1000)
    assert h.count() == 2


# ---------------------------------------------------------------------
# count / reset
# ---------------------------------------------------------------------

def test_count_tracks_records():
    h = HdrHistogram(1000)
    assert h.count() == 0
    for i in range(10):
        h.record(i)
    assert h.count() == 10


def test_reset_clears_count_and_counts_array():
    h = HdrHistogram(1000)
    for i in range(10):
        h.record(i)
    assert h.count() == 10
    h.reset()
    assert h.count() == 0
    for c in h.counts:
        assert c == 0


def test_reset_allows_reuse():
    h = HdrHistogram(1000)
    h.record(500)
    h.reset()
    h.record(10)
    h.record(20)
    assert h.count() == 2
    assert h.percentile(50) >= 10


# ---------------------------------------------------------------------
# empty histogram behavior
# ---------------------------------------------------------------------

def test_percentile_raises_when_empty():
    h = HdrHistogram(1000)
    try:
        h.percentile(50)
        assert False, "expected HdrEmptyHistogramError"
    except HdrEmptyHistogramError:
        pass


def test_percentiles_raises_when_empty():
    h = HdrHistogram(1000)
    try:
        h.percentiles([50, 90])
        assert False, "expected HdrEmptyHistogramError"
    except HdrEmptyHistogramError:
        pass


def test_mean_raises_when_empty():
    h = HdrHistogram(1000)
    try:
        h.mean()
        assert False, "expected HdrEmptyHistogramError"
    except HdrEmptyHistogramError:
        pass


def test_percentile_raises_after_reset():
    h = HdrHistogram(1000)
    h.record(5)
    h.reset()
    try:
        h.percentile(50)
        assert False, "expected HdrEmptyHistogramError"
    except HdrEmptyHistogramError:
        pass


# ---------------------------------------------------------------------
# correctness: percentile / percentiles / mean vs brute force
# ---------------------------------------------------------------------

def test_percentile_matches_brute_force():
    vals = _make_sample_dataset(n=5000, seed=1)
    h = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals:
        h.record(v)
    vals.sort()

    for p in [1, 25, 50, 75, 90, 99, 99.9, 99.99, 100]:
        expected = _brute_percentile(vals, p)
        actual = h.percentile(p)
        assert actual == expected, "p%s expected %s got %s" % (p, expected, actual)


def test_percentiles_returns_list_in_input_order():
    h = HdrHistogram(1000)
    for i in range(1, 101):
        h.record(i)

    ps = [90, 10, 50]  # deliberately unsorted
    result = h.percentiles(ps)
    assert isinstance(result, list)
    assert len(result) == len(ps)
    assert result[0] == h.percentile(90)
    assert result[1] == h.percentile(10)
    assert result[2] == h.percentile(50)


def test_percentiles_handles_duplicate_requests():
    h = HdrHistogram(1000)
    for i in range(1, 101):
        h.record(i)

    ps = [50, 50, 99]
    result = h.percentiles(ps)
    assert len(result) == 3
    assert result[0] == result[1] == h.percentile(50)
    assert result[2] == h.percentile(99)


def test_percentiles_batch_matches_single_calls():
    vals = _make_sample_dataset(n=5000, seed=2)
    h = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals:
        h.record(v)

    ps = [1, 25, 50, 75, 90, 99, 99.9, 99.99, 100]
    batch = h.percentiles(ps)
    for i, p in enumerate(ps):
        assert batch[i] == h.percentile(p), "mismatch at p%s" % p


def test_percentiles_matches_brute_force():
    vals = _make_sample_dataset(n=5000, seed=3)
    h = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals:
        h.record(v)
    vals.sort()

    ps = [1, 25, 50, 75, 90, 99, 99.9, 99.99, 100]
    batch = h.percentiles(ps)
    for i, p in enumerate(ps):
        expected = _brute_percentile(vals, p)
        assert batch[i] == expected, "p%s expected %s got %s" % (p, expected, batch[i])


def test_mean_close_to_brute_force():
    vals = _make_sample_dataset(n=5000, seed=4)
    h = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals:
        h.record(v)

    expected = sum(vals) / len(vals)
    actual = h.mean()
    # HDR is lossy by design; must be within ~1% relative error at these
    # magnitudes for sig_digits=3
    rel_err = abs(actual - expected) / expected
    assert rel_err < 0.01, "mean rel error too high: %s" % rel_err


def test_total_count_matches_iter_values_sum():
    vals = _make_sample_dataset(n=3000, seed=5)
    h = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals:
        h.record(v)

    iter_total = 0
    for value, count in h._iter_values():
        iter_total += count
    assert iter_total == h.count()


# ---------------------------------------------------------------------
# merge()
# ---------------------------------------------------------------------

def test_merge_combines_counts():
    h1 = HdrHistogram(1000)
    h2 = HdrHistogram(1000)
    for i in range(10):
        h1.record(i)
    for i in range(10, 20):
        h2.record(i)

    h1.merge(h2)
    assert h1.count() == 20
    assert h1.percentile(100) == 19


def test_merge_matches_combined_direct_recording():
    vals_a = _make_sample_dataset(n=2000, seed=10)
    vals_b = _make_sample_dataset(n=2000, seed=11)

    ha = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    hb = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals_a:
        ha.record(v)
    for v in vals_b:
        hb.record(v)
    ha.merge(hb)

    combined = HdrHistogram(3_600_000, sig_digits=3, bits=32)
    for v in vals_a + vals_b:
        combined.record(v)

    for p in [50, 90, 99, 99.9]:
        assert ha.percentile(p) == combined.percentile(p), "mismatch at p%s" % p
    assert ha.count() == combined.count()


# ---------------------------------------------------------------------
# 16-bit saturation
# ---------------------------------------------------------------------

def test_16bit_bucket_saturates_without_error():
    h = HdrHistogram(1000, bits=16)
    # push a single bucket well past 65535
    for _ in range(70000):
        h.record(1)
    assert h.count() == 70000
    # bucket itself should have saturated at max_count, not wrapped/crashed
    assert max(h.counts) == h.max_count


# ---------------------------------------------------------------------
# memory_bytes()
# ---------------------------------------------------------------------

def test_memory_bytes_matches_typecode():
    h16 = HdrHistogram(1000, bits=16)
    h32 = HdrHistogram(1000, bits=32)
    assert h16.memory_bytes() == 2 * h16.counts_len
    assert h32.memory_bytes() == 4 * h32.counts_len


# ---------------------------------------------------------------------
# test runner (no pytest)
# ---------------------------------------------------------------------

def run_all_tests():
    test_names = sorted(
        name for name in globals()
        if name.startswith("test_") and callable(globals()[name])
    )

    passed = 0
    failed = 0
    failures = []

    for name in test_names:
        fn = globals()[name]
        try:
            fn()
            passed += 1
            print("PASS", name)
        except Exception as e:
            failed += 1
            failures.append((name, e))
            print("FAIL", name, "-", e)

    print("")
    print("%d passed, %d failed, %d total" % (passed, failed, passed + failed))

    if failures:
        print("")
        print("Failures:")
        for name, e in failures:
            print(" -", name, ":", e)

    return failed


if __name__ == "__main__":
    import sys
    failed_count = run_all_tests()
    sys.exit(1 if failed_count else 0)
