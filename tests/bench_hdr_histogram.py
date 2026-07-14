"""
Benchmark for HdrHistogram.record() -- the hot path.

Runs under MicroPython natively (time.ticks_us/ticks_diff) and falls back
to time.perf_counter() on CPython. Distributions are generator functions,
not precomputed lists -- materializing NUM_VALUES_PER_DISTRIBUTION ints
up front can itself exhaust memory on a microcontroller. Instead each
distribution is reseeded and regenerated identically for the baseline
pass and the record() pass, so RNG/generator overhead is common to both
and cancels out when we subtract baseline from the record() timing.

Times a no-op baseline loop (same generator, same iteration, no work)
to subtract pure interpreter/generator overhead, and reports average
per-call cost of record() for several distributions -- including some
pushed toward large values, since _get_bucket_index's loop grows with
value magnitude.

Usage:
    python3 bench_record.py
    micropython bench_record.py
"""

import random
import gc

from hdr_histogram import HdrHistogram

try:
    import time
    _ticks_us = time.ticks_us
    _ticks_diff = time.ticks_diff

    def now_us():
        return _ticks_us()

    def elapsed_us(start, end):
        return _ticks_diff(end, start)

except AttributeError:
    # CPython: no ticks_us/ticks_diff, use perf_counter (seconds, float)
    import time

    def now_us():
        return time.perf_counter() * 1_000_000.0

    def elapsed_us(start, end):
        return end - start


MAX_VALUE = 1_000_000_000  # 1ns..1sec in nanoseconds
# MAX_VALUE deliberately kept under 2**31 (~2.1e9)
# some MicroPython builds only support 31 bit native integers
# Otherwise raises OverFlowError for things like random.randint() and inside
# HdrHistogram's own bucket-index bit shifts for large max_value
NUM_VALUES_PER_DISTRIBUTION = 2000
REPEATS = 3                # timed repeats per distribution, report best
DISTRIBUTION_SEED = 1234   # reseeded before every pass so generators
                            # reproduce the identical value sequence
                            # without storing it


def gen_typical_latency():
    # most server latencies: tight cluster around a small value, in ns
    # e.g. ~200us-1ms typical, values in [50_000, 2_000_000] ns
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        yield random.randint(50_000, 2_000_000)


def gen_typical_with_tail():
    # 99.9% small, 0.1% large tail (the realistic case this data
    # structure is built for)
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        if random.random() < 0.001:
            yield random.randint(100_000_000, MAX_VALUE)  # slow tail
        else:
            yield random.randint(50_000, 2_000_000)


def gen_bimodal():
    # e.g. cache hits (fast) vs cache misses (slow), roughly 80/20 split
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        if random.random() < 0.2:
            yield random.randint(10_000_000, 100_000_000)  # miss
        else:
            yield random.randint(20_000, 500_000)  # hit


def gen_all_large():
    # pushed to the top of the range, to measure worst-case
    # _get_bucket_index loop length (most doublings needed)
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        yield random.randint(200_000_000, MAX_VALUE)


def gen_all_small():
    # pushed to the bottom of the range, minimal bucket-search loop
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        yield random.randint(1, 1000)


def gen_log_uniform():
    # values spread evenly across orders of magnitude -- exercises every
    # bucket roughly equally, good general-purpose stress case
    for _ in range(NUM_VALUES_PER_DISTRIBUTION):
        exponent = random.uniform(0, 9)  # 10^0 .. 10^9
        yield min(int(10 ** exponent), MAX_VALUE)


DISTRIBUTIONS = [
    ("typical_latency", gen_typical_latency),
    ("typical_with_tail", gen_typical_with_tail),
    ("bimodal_hit_miss", gen_bimodal),
    ("all_large_values", gen_all_large),
    ("all_small_values", gen_all_small),
    ("log_uniform_spread", gen_log_uniform),
]


def time_noop_baseline(gen_fn):
    best = None
    for _ in range(REPEATS):
        random.seed(DISTRIBUTION_SEED)
        t0 = now_us()
        for v in gen_fn():
            pass
        t1 = now_us()
        dt = elapsed_us(t0, t1)
        if best is None or dt < best:
            best = dt
    return best


def time_record(h, gen_fn):
    best = None
    for _ in range(REPEATS):
        random.seed(DISTRIBUTION_SEED)
        h.reset()
        t0 = now_us()
        for v in gen_fn():
            h.record(v)
        t1 = now_us()
        dt = elapsed_us(t0, t1)
        if best is None or dt < best:
            best = dt
    return best


def run_benchmark():
    print(
        "values per distribution:", NUM_VALUES_PER_DISTRIBUTION,
        " repeats:", REPEATS,
    )
    print("")
    header = "%-20s %12s %12s %12s" % (
        "distribution", "baseline_us", "record_us", "record_us/call"
    )
    print(header)
    print("-" * len(header))

    for bits in (16, 32):
        # try reduce memory pressure
        gc.collect()

        print("")
        print("--- bits=%d ---" % bits)
        h = HdrHistogram(MAX_VALUE, sig_digits=3, bits=bits)
        print(h.memory_bytes())

        for name, gen_fn in DISTRIBUTIONS:
            baseline_us = time_noop_baseline(gen_fn)
            record_total_us = time_record(h, gen_fn)
            net_us = record_total_us - baseline_us
            per_call_us = net_us / float(NUM_VALUES_PER_DISTRIBUTION)

            print(
                "%-20s %12.1f %12.1f %12.4f"
                % (name, baseline_us, record_total_us, per_call_us)
            )


if __name__ == "__main__":
    run_benchmark()
