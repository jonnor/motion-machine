"""
test_tscompress.py - Test suite for tscompress.py
"""
import sys
sys.path.insert(0, '/mnt/user-data/outputs')

import array
import os
import delta_simple9 as tc

PATH = '/tmp/ds9_test.des9'

def _read_all(path):
    """Read all rows from a file into a single row-major array.array('h')."""
    out = array.array('h')
    with tc.Reader(path) as r:
        buf = array.array('h', [0] * (r.chunk_size * r.n_cols))
        while True:
            n = r.read_chunk_into(buf)
            if n == 0:
                break
            out.extend(buf[:n * r.n_cols])
    return out

def _read_chunks(path):
    """Read all chunks, return list of array.array('h') per chunk."""
    chunks = []
    with tc.Reader(path) as r:
        buf = array.array('h', [0] * (r.chunk_size * r.n_cols))
        while True:
            n = r.read_chunk_into(buf)
            if n == 0:
                break
            chunks.append(array.array('h', buf[:n * r.n_cols]))
    return chunks

def remove():
    try: os.remove(PATH)
    except OSError: pass

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)

# ---------------------------------------------------------------------------

def test_roundtrip_single_col():
    remove()
    with tc.Writer(PATH, n_cols=1, chunk_size=64) as w:
        for i in range(64): w.write_row([i])
    out = _read_all(PATH)
    check(len(out) == 64, f"len={len(out)}")
    for i in range(64):
        check(out[i] == i, f"row {i}: {out[i]}")

def test_roundtrip_multi_col():
    remove()
    rows = [(i, i*2 - 100, -i + 50) for i in range(300)]
    with tc.Writer(PATH, n_cols=3, chunk_size=128) as w:
        for row in rows: w.write_row(row)
    data = _read_all(PATH)
    check(len(data)//3 == 300, f"nrows={len(data)//3}")
    for i, row in enumerate(rows):
        got = (data[i*3], data[i*3+1], data[i*3+2])
        check(got == row, f"row {i}: {row} != {got}")

def test_partial_chunk():
    """Last chunk has fewer rows than chunk_size."""
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=100) as w:
        for i in range(250): w.write_row((i, -i))
    buf = array.array('h', [0] * 200)
    collected = []
    with tc.Reader(PATH) as r:
        while True:
            n = r.read_chunk_into(buf)
            if n == 0: break
            for i in range(n): collected.append((buf[i*2], buf[i*2+1]))
    check(len(collected) == 250, f"nrows={len(collected)}")
    for i, got in enumerate(collected):
        check(got == (i, -i), f"row {i}: {got}")

def test_exactly_one_chunk():
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(64): w.write_row((i, -i))
    out = _read_all(PATH)
    check(len(out)//2 == 64, f"nrows={len(out)//2}")
    for i in range(64):
        check(out[i*2] == i and out[i*2+1] == -i, f"row {i}")

def test_single_row():
    remove()
    with tc.Writer(PATH, n_cols=3, chunk_size=64) as w:
        w.write_row((7, -3, 100))
    out = _read_all(PATH)
    check(len(out) == 3, f"len={len(out)}")
    check((out[0], out[1], out[2]) == (7, -3, 100), f"{list(out)}")

def test_int16_extremes():
    """Values at the int16 boundary, including worst-case delta."""
    remove()
    vals = [32767, -32768, 0, -1, 1, 32767, -32768]
    with tc.Writer(PATH, n_cols=1, chunk_size=64) as w:
        for v in vals: w.write_row([v])
    out = _read_all(PATH)
    check(list(out) == vals, f"{list(out)}")

def test_write_rows_bulk():
    """write_rows from a row-major array.array."""
    remove()
    bulk = array.array('h')
    for i in range(80): bulk.extend(array.array('h', [i, -i]))
    with tc.Writer(PATH, n_cols=2, chunk_size=32) as w:
        w.write_rows(bulk)
    out = _read_all(PATH)
    check(len(out)//2 == 80, f"nrows={len(out)//2}")
    for i in range(80):
        check(out[i*2] == i and out[i*2+1] == -i, f"row {i}")

def test_read_chunk_into():
    """read_chunk_into writes correct strided data into caller's buffer."""
    remove()
    with tc.Writer(PATH, n_cols=3, chunk_size=50) as w:
        for i in range(100): w.write_row((i, i+1, i+2))
    buf = array.array('h', [0] * (50 * 3))
    chunks = []
    with tc.Reader(PATH) as r:
        while True:
            n = r.read_chunk_into(buf)
            if n == 0: break
            chunks.append(list(buf[:n*3]))
    check(len(chunks) == 2, f"nchunks={len(chunks)}")
    for ci, chunk in enumerate(chunks):
        for r in range(len(chunk)//3):
            i = ci*50 + r
            check(chunk[r*3] == i and chunk[r*3+1] == i+1 and chunk[r*3+2] == i+2,
                  f"chunk {ci} row {r}")

def test_append_same_session():
    """Flush multiple times within one open writer (multi-chunk session)."""
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(200): w.write_row((i, -i))
    out = _read_all(PATH)
    check(len(out)//2 == 200, f"nrows={len(out)//2}")
    for i in range(200):
        check(out[i*2] == i and out[i*2+1] == -i, f"row {i}")

def test_append_reopen():
    """Open an existing file and append more data."""
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(128): w.write_row((i, -i))
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(128, 256): w.write_row((i, -i))
    out = _read_all(PATH)
    check(len(out)//2 == 256, f"nrows={len(out)//2}")
    for i in range(256):
        check(out[i*2] == i and out[i*2+1] == -i, f"row {i}")

def test_append_reopen_partial_chunk():
    """Append to a file whose last chunk was partial."""
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(100): w.write_row((i, -i))   # 1 full + 1 partial chunk
    with tc.Writer(PATH, n_cols=2, chunk_size=64) as w:
        for i in range(100, 200): w.write_row((i, -i))
    out = _read_all(PATH)
    check(len(out)//2 == 200, f"nrows={len(out)//2}")
    for i in range(200):
        check(out[i*2] == i and out[i*2+1] == -i, f"row {i}")

def test_append_reopen_multiple_times():
    """Append in three separate sessions."""
    remove()
    for session in range(3):
        with tc.Writer(PATH, n_cols=1, chunk_size=32) as w:
            for i in range(session*50, session*50 + 50):
                w.write_row([i])
    out = _read_all(PATH)
    check(len(out) == 150, f"len={len(out)}")
    for i in range(150):
        check(out[i] == i, f"row {i}: {out[i]}")

def test_append_header_mismatch_rejected():
    """Reopening with wrong n_cols or chunk_size raises ValueError."""
    remove()
    with tc.Writer(PATH, n_cols=2, chunk_size=64): pass
    try:
        tc.Writer(PATH, n_cols=3, chunk_size=64)
        check(False, "should have raised on n_cols mismatch")
    except ValueError:
        pass
    try:
        tc.Writer(PATH, n_cols=2, chunk_size=32)
        check(False, "should have raised on chunk_size mismatch")
    except ValueError:
        pass

def test_compression_ratio():
    """Incrementing data should compress significantly."""
    remove()
    with tc.Writer(PATH, n_cols=3, chunk_size=128) as w:
        for i in range(300): w.write_row((i, i*2-100, -i+50))
    comp = os.stat(PATH)[6]
    raw  = 300 * 3 * 2
    ratio = raw / comp
    check(ratio > 3.0, f"compression ratio too low: {ratio:.2f}x")

# ---------------------------------------------------------------------------

TESTS = [
    test_roundtrip_single_col,
    test_roundtrip_multi_col,
    test_partial_chunk,
    test_exactly_one_chunk,
    test_single_row,
    test_int16_extremes,
    test_write_rows_bulk,
    test_read_chunk_into,
    test_append_same_session,
    test_append_reopen,
    test_append_reopen_partial_chunk,
    test_append_reopen_multiple_times,
    test_append_header_mismatch_rejected,
    test_compression_ratio,
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
remove()

print(f"\n{passed}/{passed+failed} passed")
if failed:
    sys.exit(1)
