"""
realworld_test.py - Real-world scenario test for MicroHive

Scenario:
  - 12 hours of hourly-partitioned data
  - 1-second resolution, 2 columns (sensor_a, sensor_b)
  - Synthetic signal: 20-minute periodic pattern + noise + random walk
  - int16 throughout

Steps:
  1. Stream-generate full 12h time series to source.npy
  2. Stream-load source.npy into MicroHive via append_data
  3. Query middle 2 hours, save result to query_result.npy

Run:
    python3 realworld_test.py          # CPython (host dev)
    micropython realworld_test.py      # MicroPython Unix port
    mpremote run realworld_test.py     # ESP32
"""

import array
import math
import os
import sys
import time

import npyfile
from microhive import MicroHive, _makedirs, TYPECODE

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.time() * 1000)

def _diff_ms(t0):
    try:
        return time.ticks_diff(time.ticks_ms(), t0)
    except AttributeError:
        return int(time.time() * 1000) - t0

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _rmdir_recursive(path):
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for e in entries:
        full = "{}/{}".format(path, e)
        try:
            os.listdir(full)
            _rmdir_recursive(full)
            os.rmdir(full)
        except OSError:
            try:
                os.remove(full)
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Time patching (simulate wall clock advancing during load)
# On a real device time.time() advances naturally; here we control it so
# append_data routes rows to the correct hour partition.
# ---------------------------------------------------------------------------

class _TimePatch:
    def __init__(self):
        self._real_time = None
        self._value = 0

    def patch(self, epoch_s):
        self._value = epoch_s
        if self._real_time is None:
            self._real_time = time.time
            time.time = lambda: self._value

    def restore(self):
        if self._real_time is not None:
            time.time = self._real_time
            self._real_time = None

# ---------------------------------------------------------------------------
# Synthetic signal generator (no global state)
# ---------------------------------------------------------------------------

class _SignalGen:
    """
    Two-channel synthetic time series:
      sensor_a: sine with 20-min period + noise + slow random walk
      sensor_b: cosine (phase-shifted pi/3) + independent noise + random walk
    All values stay within int16 range.
    """

    _PI2 = 2.0 * 3.141592653589793

    def __init__(self, period_s=1200, amplitude=8000, noise_scale=400,
                 walk_scale=20, walk_limit=4000, seed=12345):
        self._period_s    = period_s
        self._amplitude   = amplitude
        self._noise_scale = noise_scale
        self._walk_scale  = walk_scale
        self._walk_limit  = walk_limit
        self._lcg         = seed
        self._walk_a      = 0
        self._walk_b      = 0

    def _lcg_next(self):
        self._lcg = (self._lcg * 1664525 + 1013904223) & 0xFFFFFFFF
        return self._lcg

    def _noise(self, scale):
        """Approximate zero-mean noise via sum of two uniform LCG draws."""
        r1 = self._lcg_next()
        r2 = self._lcg_next()
        v = ((r1 & 0xFFFF) + (r2 & 0xFFFF)) - 0xFFFF
        return (v * scale) >> 16

    def sample(self, t):
        """Return (a, b) int16 sample at time offset t (seconds)."""
        angle  = self._PI2 * t / self._period_s
        base_a = int(self._amplitude * math.sin(angle))
        base_b = int(self._amplitude * math.cos(angle + 1.047197551))  # pi/3 phase
        lim    = self._walk_limit
        self._walk_a = max(-lim, min(lim, self._walk_a + self._noise(self._walk_scale)))
        self._walk_b = max(-lim, min(lim, self._walk_b + self._noise(self._walk_scale)))
        a = max(-32768, min(32767, base_a + self._noise(self._noise_scale) + self._walk_a))
        b = max(-32768, min(32767, base_b + self._noise(self._noise_scale) + self._walk_b))
        return a, b

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _step_generate(source_npy, total_rows, n_cols, signal):
    """Step 1: Stream-generate synthetic time series to source.npy."""
    print("Step 1: Generate source.npy ({} rows, {} cols)".format(total_rows, n_cols))
    t0 = _ms()

    chunk_rows = 256
    buf = array.array(TYPECODE, [0] * (chunk_rows * n_cols))

    with npyfile.Writer(source_npy, shape=(total_rows, n_cols), typecode=TYPECODE) as w:
        row = 0
        while row < total_rows:
            n = min(chunk_rows, total_rows - row)
            for i in range(n):
                a, b = signal.sample(row + i)
                buf[i * n_cols]     = a
                buf[i * n_cols + 1] = b
            if n == chunk_rows:
                print('write chunk')
                w.write_values(buf, typecode=TYPECODE)
            else:
                w.write_values(array.array(TYPECODE, buf[:n * n_cols]), typecode=TYPECODE)
            row += n

    size_bytes = os.stat(source_npy)[6]
    print("  Done in {} ms  |  file: {} bytes ({} KB)".format(
        _diff_ms(t0), size_bytes, size_bytes // 1024))
    print()


def _step_load(source_npy, base_dir, resource, n_cols, append_chunk_rows,
               base_epoch, time_patch):
    """Step 2: Stream source.npy into MicroHive via append_data."""
    print("Step 2: Load source.npy into MicroHive via append_data")
    print("  Chunk: {} rows/call  |  Granularity: {}".format(
        append_chunk_rows, resource['granularity']))
    t0 = _ms()

    db = MicroHive(base_dir, {"sensor": resource})
    latencies = []

    with npyfile.Reader(source_npy) as reader:
        buf      = array.array(TYPECODE)
        buf_rows = 0
        row_idx  = 0

        for chunk in reader.read_data_chunks(n_cols):
            buf.extend(chunk)
            buf_rows += 1
            row_idx  += 1

            if buf_rows >= append_chunk_rows:
                time_patch.patch(base_epoch + (row_idx - buf_rows))
                t_a = _ms()
                db.append_data("sensor", buf)
                latencies.append(_diff_ms(t_a))
                buf      = array.array(TYPECODE)
                buf_rows = 0

        if buf_rows > 0:
            time_patch.patch(base_epoch + (row_idx - buf_rows))
            t_a = _ms()
            db.append_data("sensor", buf)
            latencies.append(_diff_ms(t_a))

    time_patch.restore()

    max_ms  = max(latencies)
    avg_ms  = sum(latencies) // len(latencies)
    over100 = sum(1 for l in latencies if l > 100)

    print("  Total appends: {}  |  total rows: {}".format(len(latencies), row_idx))
    print("  Total time: {} ms".format(_diff_ms(t0)))
    print("  Avg / max append latency: {} ms / {} ms  [{}]".format(
        avg_ms, max_ms, "OK" if max_ms <= 100 else "FAIL >100ms"))
    print("  Appends over 100ms: {} / {}".format(over100, len(latencies)))
    print("  Partitions created: {}".format(db.get_info("sensor")['n_partitions']))
    print()
    return db


def _step_query(db, result_npy, n_cols, base_epoch,
                query_start_offset_h, query_end_offset_h):
    """Step 3: Query a middle window and stream result to query_result.npy."""
    start_s       = base_epoch + query_start_offset_h * 3600
    end_s         = base_epoch + query_end_offset_h * 3600
    expected_rows = (query_end_offset_h - query_start_offset_h) * 3600

    print("Step 3: Query hours {}..{} -> query_result.npy".format(
        query_start_offset_h, query_end_offset_h))
    print("  Expected rows: {}".format(expected_rows))
    t0 = _ms()

    # Pass 1: count rows (needed to write .npy shape header up front)
    count = sum(
        len(chunk) // n_cols
        for chunk in db.get_timerange("sensor", start_s, end_s, chunk_rows=3600)
    )

    # Pass 2: stream into output file
    total_rows   = 0
    max_chunk_ms = 0
    with npyfile.Writer(result_npy, shape=(count, n_cols), typecode=TYPECODE) as w:
        t_chunk = _ms()
        for chunk in db.get_timerange("sensor", start_s, end_s, chunk_rows=3600):
            max_chunk_ms = max(max_chunk_ms, _diff_ms(t_chunk))
            total_rows  += len(chunk) // n_cols
            w.write_values(chunk, typecode=TYPECODE)
            t_chunk = _ms()

    size_bytes = os.stat(result_npy)[6]
    print("  Rows returned: {}  [{}]".format(
        total_rows, "OK" if total_rows == expected_rows else "FAIL"))
    print("  Max chunk latency: {} ms  [{}]".format(
        max_chunk_ms, "OK" if max_chunk_ms <= 100 else "FAIL >100ms"))
    print("  Total query time: {} ms".format(_diff_ms(t0)))
    print("  Output file: {} bytes ({} KB)".format(size_bytes, size_bytes // 1024))
    print()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_12h_hourly_periodic_signal():
    base_dir   = "/tmp/microhive_rw" if sys.platform != "esp32" else "/mw_rw"
    source_npy = base_dir + "/source.npy"
    result_npy = base_dir + "/query_result.npy"

    duration_h        = 12
    n_cols            = 2
    hop_us            = 1_000_000   # 1 second
    append_chunk_rows = 60          # 1 minute of data per append call
    total_rows        = duration_h * 3600

    try:
        base_epoch = int(time.mktime((2025, 6, 1, 0, 0, 0, 0, 0)))
    except TypeError:
        base_epoch = int(time.mktime((2025, 6, 1, 0, 0, 0, 0, 0, -1)))

    resource = {
        "hop": hop_us,
        "columns": ["sensor_a", "sensor_b"],
        "dtype": "int16",
        "granularity": "hour",
    }

    print("=== test_12h_hourly_periodic_signal ===")
    print("Platform: {}  |  Base dir: {}".format(sys.platform, base_dir))
    print("{} hours  |  {} rows  |  1s hop  |  {} cols".format(
        duration_h, total_rows, n_cols))
    print()

    _rmdir_recursive(base_dir)
    _makedirs(base_dir)

    _step_generate(source_npy, total_rows, n_cols, _SignalGen())
    db = _step_load(source_npy, base_dir, resource, n_cols,
                    append_chunk_rows, base_epoch, _TimePatch())
    _step_query(db, result_npy, n_cols, base_epoch,
                query_start_offset_h=5, query_end_offset_h=7)

    print("Files written:")
    print("  {}".format(source_npy))
    print("  {}".format(result_npy))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_12h_hourly_periodic_signal()
