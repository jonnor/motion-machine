"""
bench_microhive.py - Benchmark append and query performance of MicroHive.

Run with:
    micropython bench_microhive.py
    python3 bench_microhive.py
    mpremote run firmware/bench_microhive.py
"""

import sys, array, os, time
sys.path.insert(0, '.')
import microhive as mh

# ---------------------------------------------------------------------------
def _ms():
    try: return time.ticks_ms()
    except AttributeError: return int(time.time() * 1000)

def _diff(t):
    try: return time.ticks_diff(time.ticks_ms(), t)
    except AttributeError: return int(time.time() * 1000) - t

def _rmdir(p):
    try:
        for e in os.listdir(p):
            f = p + '/' + e
            try: os.listdir(f); _rmdir(f); os.rmdir(f)
            except:
                try: os.remove(f)
                except: pass
        os.rmdir(p)
    except: pass

# ---------------------------------------------------------------------------
BASE    = '/tmp/mh_bench' if sys.platform != 'esp32' else '/bench'
N_COLS  = 7
HOP_US  = 1_000_000   # 1Hz
GRAN    = 'hour'
CHUNK   = 100          # rows per append_data call (realistic streaming size)

RESOURCE = {
    'hop':         HOP_US,
    'columns':     ['c{}'.format(i) for i in range(N_COLS)],
    'dtype':       'int16',
    'granularity': GRAN,
}

def make_db():
    return mh.MicroHive(BASE, {'s': RESOURCE}, debug=False)

def make_data(n, offset=0):
    d = array.array('h')
    for i in range(n):
        for c in range(N_COLS):
            d.append((offset + i + c * 100) % 30000)
    return d

# ---------------------------------------------------------------------------
def bench_append(n_hours, chunk_rows):
    """Append n_hours of data in chunk_rows-sized calls. Measures throughput."""
    _rmdir(BASE); mh._makedirs(BASE)
    t0  = 1735689600  # 2025-01-01 00:00:00 UTC as Unix epoch seconds
    total_rows = n_hours * 3600

    chunk = make_data(chunk_rows)

    t = _ms()
    with make_db() as db:
        for start in range(0, total_rows, chunk_rows):
            actual = min(chunk_rows, total_rows - start)
            db.append_data('s', chunk if actual == chunk_rows else make_data(actual),
                           timestamp_s=t0 + start)
    elapsed = _diff(t)

    n_calls = (total_rows + chunk_rows - 1) // chunk_rows
    rows_ms = total_rows / elapsed if elapsed else 0
    ms_call = elapsed / n_calls if n_calls else 0
    print("  append {}h x{}cols chunk={}: {}ms total  {:.1f} rows/ms  {:.1f} ms/call".format(
        n_hours, N_COLS, chunk_rows, elapsed, rows_ms, ms_call))
    return db, t0, total_rows

def bench_query(db, t0, total_rows, chunk_rows=256):
    """Query all data back. Measures read throughput."""
    t = _ms()
    rows = 0
    for chunk in db.get_timerange('s', t0, t0 + total_rows, chunk_rows=chunk_rows):
        rows += len(chunk) // N_COLS
    elapsed = _diff(t)
    rows_ms = rows / elapsed if elapsed else 0
    print("  query  {}rows chunk={}: {}ms  {:.1f} rows/ms  {}".format(
        rows, chunk_rows, elapsed, rows_ms,
        'OK' if rows == total_rows else 'WRONG ({})'.format(rows)))

# ---------------------------------------------------------------------------
print("MicroHive benchmark  cols={}  hop={}us  gran={}".format(N_COLS, HOP_US, GRAN))
print("platform:", sys.platform)
print()

# Raw filesystem benchmark
print("--- raw filesystem ---")
import struct
test_path = BASE + '/_io_test.bin'
mh._makedirs(BASE)
data_1kb = bytes(range(256)) * 4
t = _ms()
for _ in range(50):
    f = open(test_path, 'wb')
    f.write(data_1kb)
    f.flush()
    f.close()
t_write = _diff(t)
t = _ms()
for _ in range(50):
    f = open(test_path, 'rb')
    f.read()
    f.close()
t_read = _diff(t)
try: os.remove(test_path)
except: pass
print("  open+write+flush+close 1KB x50: {}ms  ({:.1f} ms/op)".format(t_write, t_write/50))
print("  open+read+close 1KB x50:        {}ms  ({:.1f} ms/op)".format(t_read, t_read/50))
print()

print("--- append ---")
db, t0, n = bench_append(n_hours=1, chunk_rows=100)
db, t0, n = bench_append(n_hours=1, chunk_rows=1000)

print()
print("--- query (after 1h write, chunk=100) ---")
db, t0, n = bench_append(n_hours=1, chunk_rows=100)
bench_query(db, t0, n, chunk_rows=64)
bench_query(db, t0, n, chunk_rows=256)
bench_query(db, t0, n, chunk_rows=512)

print()
_rmdir(BASE)
print("done")
