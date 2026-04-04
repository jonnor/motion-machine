"""
test_microhive.py - Test suite for microhive.py (delta_simple9 backend)

Run with:
    micropython test_microhive.py
    python3 test_microhive.py
"""

import array
import math
import os
import sys
import time

sys.path.insert(0, '.')

import microhive as mh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = '/tmp/mh_test'

def _ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.time() * 1000)

def _diff(t0):
    try:
        return time.ticks_diff(time.ticks_ms(), t0)
    except AttributeError:
        return int(time.time() * 1000) - t0

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

def setup():
    _rmdir_recursive(BASE)
    mh._makedirs(BASE)
    try:
        import gc; gc.collect()
    except: pass

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)

def _base_epoch():
    """Fixed epoch: 2025-01-01 00:00:00 as Unix epoch seconds.
    The public API of microhive takes Unix epoch seconds."""
    return mh._parts_to_epoch(2025, 1, 1, 0, 0, 0)

def _collect(db, resource, start_s, end_s, chunk_rows=64):
    """Collect all rows from get_timerange into a single array."""
    out = array.array('h')
    for chunk in db.get_timerange(resource, start_s, end_s, chunk_rows=chunk_rows):
        out.extend(chunk)
    return out

def _make_db(granularity='hour', hop_us=1_000_000, n_cols=2, debug=False):
    return mh.MicroHive(BASE, {
        'sensor': {
            'hop': hop_us,
            'columns': ['a', 'b'][:n_cols],
            'dtype': 'int16',
            'granularity': granularity,
        }
    }, debug=debug)

def _rows(n, n_cols=2, offset=0):
    """Generate n rows of recognisable test data as array.array('h')."""
    data = array.array('h')
    for i in range(n):
        for c in range(n_cols):
            data.append((offset + i + c * 100) % 30000)
    return data

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_append_and_query():
    """Basic: write 10 rows, read them back."""
    setup()
    t0 = _base_epoch()
    db = _make_db()
    data = _rows(10)
    db.append_data('sensor', data, timestamp_s=t0)
    out = _collect(db, 'sensor', t0, t0 + 10)
    check(len(out) == 20, "expected 20 items, got {}".format(len(out)))
    for i in range(20):
        check(out[i] == data[i], "item {}: {} != {}".format(i, out[i], data[i]))

def test_query_full_partition():
    """Write exactly one hour of 1Hz data, query it all back."""
    setup()
    t0   = _base_epoch()
    db   = _make_db(granularity='hour', hop_us=1_000_000)
    n    = 3600
    data = _rows(n)
    db.append_data('sensor', data, timestamp_s=t0)
    # Verify chunk-by-chunk to avoid accumulating 14400 bytes on embedded targets
    row_offset = 0
    for chunk in db.get_timerange('sensor', t0, t0 + n, chunk_rows=256):
        chunk_n = len(chunk) // 2
        for r in range(chunk_n):
            i = row_offset + r
            check(chunk[r*2]   == data[i*2],   "row {} col 0: {} != {}".format(i, chunk[r*2],   data[i*2]))
            check(chunk[r*2+1] == data[i*2+1], "row {} col 1: {} != {}".format(i, chunk[r*2+1], data[i*2+1]))
        row_offset += chunk_n
    check(row_offset == n, "expected {} rows, got {}".format(n, row_offset))

def test_query_sub_range():
    """Write 1h, query only the middle portion."""
    setup()
    t0   = _base_epoch()
    db   = _make_db(granularity='hour', hop_us=1_000_000)
    n    = 3600
    data = _rows(n)
    db.append_data('sensor', data, timestamp_s=t0)

    q_start = t0 + 1000
    q_end   = t0 + 2000
    expected_rows = 1000
    row_offset = 0
    for chunk in db.get_timerange('sensor', q_start, q_end, chunk_rows=256):
        chunk_n = len(chunk) // 2
        for r in range(chunk_n):
            for c in range(2):
                got      = chunk[r * 2 + c]
                expected = data[(1000 + row_offset + r) * 2 + c]
                check(got == expected,
                      "row {} col {}: got {} expected {}".format(row_offset+r, c, got, expected))
        row_offset += chunk_n
    check(row_offset == expected_rows,
          "expected {} rows, got {}".format(expected_rows, row_offset))

def test_multi_chunk_append():
    """Multiple append calls to same partition accumulate correctly."""
    setup()
    t0   = _base_epoch()
    db   = _make_db(granularity='hour', hop_us=1_000_000)
    hop  = 100  # rows per append
    n    = 500

    all_data = array.array('h')
    for i in range(0, n, hop):
        chunk = _rows(hop, offset=i)
        db.append_data('sensor', chunk, timestamp_s=t0 + i)
        all_data.extend(chunk)

    out = _collect(db, 'sensor', t0, t0 + n)
    check(len(out) // 2 == n, "expected {} rows, got {}".format(n, len(out) // 2))
    for i in range(len(all_data)):
        check(out[i] == all_data[i], "item {}: {} != {}".format(i, out[i], all_data[i]))

def test_cross_partition_split_hour():
    """Data spanning two hourly partitions is split correctly."""
    setup()
    t0  = _base_epoch()          # start of hour 0
    t1  = t0 + 3600              # start of hour 1
    db  = _make_db(granularity='hour', hop_us=1_000_000)
    n   = 200                    # 100 rows in hour 0, 100 in hour 1
    # start 100s before boundary
    ts  = t1 - 100
    data = _rows(n)
    db.append_data('sensor', data, timestamp_s=ts)

    check(db.get_info('sensor')['n_partitions'] == 2, "expected 2 partitions")

    out = _collect(db, 'sensor', ts, ts + n, chunk_rows=64)
    check(len(out) // 2 == n,
          "expected {} rows, got {}".format(n, len(out) // 2))
    for i in range(len(data)):
        check(out[i] == data[i], "item {}: {} != {}".format(i, out[i], data[i]))

def test_cross_partition_split_minute():
    """Minute granularity: data split across 3 minute partitions."""
    setup()
    t0  = _base_epoch()
    db  = _make_db(granularity='minute', hop_us=1_000_000)
    # start 30s before end of minute 0 → spans minutes 0, 1, 2
    ts   = t0 + 30
    n    = 150
    data = _rows(n)
    db.append_data('sensor', data, timestamp_s=ts)

    check(db.get_info('sensor')['n_partitions'] == 3, "expected 3 partitions")

    out = _collect(db, 'sensor', ts, ts + n)
    check(len(out) // 2 == n,
          "expected {} rows, got {}".format(n, len(out) // 2))
    for i in range(len(data)):
        check(out[i] == data[i], "item {}: {} != {}".format(i, out[i], data[i]))

def test_query_across_partitions():
    """Query spanning multiple partitions returns contiguous data."""
    setup()
    t0  = _base_epoch()
    db  = _make_db(granularity='hour', hop_us=1_000_000)

    for h in range(4):
        db.append_data('sensor', _rows(3600, offset=h * 1000), timestamp_s=t0 + h * 3600)

    # Verify chunk-by-chunk to avoid accumulating 4*14400 bytes
    row_offset = 0
    for chunk in db.get_timerange('sensor', t0, t0 + 4 * 3600, chunk_rows=256):
        chunk_n = len(chunk) // 2
        for r in range(chunk_n):
            i      = row_offset + r
            h      = i // 3600
            ri     = i % 3600
            exp_a  = (h * 1000 + ri) % 30000
            exp_b  = (h * 1000 + ri + 100) % 30000
            check(chunk[r*2]   == exp_a, "row {} col 0: {} != {}".format(i, chunk[r*2],   exp_a))
            check(chunk[r*2+1] == exp_b, "row {} col 1: {} != {}".format(i, chunk[r*2+1], exp_b))
        row_offset += chunk_n
    check(row_offset == 4 * 3600,
          "expected {} rows, got {}".format(4 * 3600, row_offset))

def test_query_missing_partition():
    """Querying a time range with no data returns empty."""
    setup()
    t0  = _base_epoch()
    db  = _make_db()
    out = _collect(db, 'sensor', t0, t0 + 3600)
    check(len(out) == 0, "expected empty, got {} items".format(len(out)))

def test_query_partial_overlap_start():
    """Query starts before data: only overlapping rows returned."""
    setup()
    t0   = _base_epoch()
    db   = _make_db(granularity='hour', hop_us=1_000_000)
    ts   = t0 + 500   # data starts at row 500
    n    = 200
    data = _rows(n)
    db.append_data('sensor', data, timestamp_s=ts)

    # Query from t0 (before data) to ts+100 (mid-data)
    out = _collect(db, 'sensor', t0, ts + 100)
    check(len(out) // 2 == 100,
          "expected 100 rows, got {}".format(len(out) // 2))
    for r in range(100):
        for c in range(2):
            check(out[r*2+c] == data[r*2+c],
                  "row {} col {}: {} != {}".format(r, c, out[r*2+c], data[r*2+c]))

def test_int16_extremes():
    """int16 min/max values survive round-trip."""
    setup()
    t0   = _base_epoch()
    db   = _make_db(n_cols=2)
    data = array.array('h', [32767, -32768, -1, 1, 0, 0, 32767, -32768])
    db.append_data('sensor', data, timestamp_s=t0)
    out = _collect(db, 'sensor', t0, t0 + 4)
    check(list(out) == list(data),
          "extremes mismatch: {} != {}".format(list(out), list(data)))

def test_out_of_order_rejected():
    """Writing to an earlier partition after a later one raises ValueError."""
    setup()
    t0  = _base_epoch()
    db  = _make_db()
    db.append_data('sensor', _rows(10), timestamp_s=t0 + 7200)  # hour 2
    try:
        db.append_data('sensor', _rows(10), timestamp_s=t0)      # hour 0
        check(False, "should have raised ValueError")
    except ValueError:
        pass

def test_n_partitions_correct():
    """get_info returns the correct partition count."""
    setup()
    t0 = _base_epoch()
    db = _make_db(granularity='hour')
    for h in range(5):
        db.append_data('sensor', _rows(10), timestamp_s=t0 + h * 3600)
    info = db.get_info('sensor')
    check(info['n_partitions'] == 5,
          "expected 5 partitions, got {}".format(info['n_partitions']))

def test_reopen_and_append():
    """
    Append to a partition, close the MicroHive, reopen, append more.
    delta_simple9 Writer appends to existing .des9 files transparently.
    """
    setup()
    t0   = _base_epoch()
    data1 = _rows(100)
    data2 = _rows(100, offset=100)

    db1 = _make_db()
    db1.append_data('sensor', data1, timestamp_s=t0)

    db2 = _make_db()   # fresh MicroHive instance, same directory
    db2.append_data('sensor', data2, timestamp_s=t0 + 100)

    out = _collect(db2, 'sensor', t0, t0 + 200)
    all_data = array.array('h')
    all_data.extend(data1)
    all_data.extend(data2)
    check(len(out) // 2 == 200,
          "expected 200 rows, got {}".format(len(out) // 2))
    for i in range(len(all_data)):
        check(out[i] == all_data[i],
              "item {}: {} != {}".format(i, out[i], all_data[i]))

def test_12h_hourly_scenario():
    """
    Integration: 12 hours of 1Hz 2-col data, hourly partitions.
    Write in 1000-row chunks, query middle 2 hours, verify correctness.
    """
    setup()
    t0        = _base_epoch()
    hop_us    = 1_000_000
    n_cols    = 2
    total_s   = 12 * 3600
    chunk_n   = 1000

    db = mh.MicroHive(BASE, {
        'sensor': {
            'hop': hop_us,
            'columns': ['a', 'b'],
            'dtype': 'int16',
            'granularity': 'hour',
        }
    }, debug=False)

    # Use simple linear pattern so values are predictable
    all_data = array.array('h')
    for start in range(0, total_s, chunk_n):
        n    = min(chunk_n, total_s - start)
        rows = array.array('h')
        for i in range(n):
            rows.append((start + i) % 30000)       # col a
            rows.append((start + i + 1000) % 30000) # col b
        db.append_data('sensor', rows, timestamp_s=t0 + start)
        all_data.extend(rows)

    check(db.get_info('sensor')['n_partitions'] == 12, "expected 12 partitions")

    # Query hours 5-7 (rows 18000..25199)
    q_start      = t0 + 5 * 3600
    q_end        = t0 + 7 * 3600
    expected_n   = 2 * 3600
    expected_off = 5 * 3600

    t_q = _ms()
    row_offset = 0
    for chunk in db.get_timerange('sensor', q_start, q_end, chunk_rows=256):
        chunk_n = len(chunk) // 2
        for r in range(chunk_n):
            i     = row_offset + r
            exp_a = (expected_off + i) % 30000
            exp_b = (expected_off + i + 1000) % 30000
            check(chunk[r*2]   == exp_a, "row {} col 0: {} != {}".format(i, chunk[r*2],   exp_a))
            check(chunk[r*2+1] == exp_b, "row {} col 1: {} != {}".format(i, chunk[r*2+1], exp_b))
        row_offset += chunk_n
    elapsed = _diff(t_q)
    check(row_offset == expected_n,
          "expected {} rows, got {}".format(expected_n, row_offset))
    print("  12h scenario: query {} rows in {}ms".format(expected_n, elapsed))

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_single_append_and_query,
    test_query_full_partition,
    test_query_sub_range,
    test_multi_chunk_append,
    test_cross_partition_split_hour,
    test_cross_partition_split_minute,
    test_query_across_partitions,
    test_query_missing_partition,
    test_query_partial_overlap_start,
    test_int16_extremes,
    test_out_of_order_rejected,
    test_n_partitions_correct,
    test_reopen_and_append,
    test_12h_hourly_scenario,
]

passed = failed = 0
for t in TESTS:
    try:
        t()
        print("  OK:", t.__name__)
        passed += 1
    except Exception as e:
        print("FAIL:", t.__name__, "-", e)
        
        
        failed += 1

_rmdir_recursive(BASE)
print("\n{}/{} passed".format(passed, passed + failed))
if failed:
    sys.exit(1)
