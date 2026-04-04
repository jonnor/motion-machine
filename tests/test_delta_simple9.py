"""
test_tscompress.py - Test suite for tscompress.py
"""
import sys
sys.path.insert(0, '.')

import array
import os
import delta_simple9 as tc
import time

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

print("\n{}/{} passed".format(passed, passed+failed))
if failed:
    sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _ms():
    try: return time.ticks_ms()
    except AttributeError: return int(time.time() * 1000)

def _diff(t):
    try: return time.ticks_diff(time.ticks_ms(), t)
    except AttributeError: return int(time.time() * 1000) - t

def bench_encode_1000_rows():
    """Benchmark write_rows: 1000 rows, 2 cols, repeated 20x."""
    d = array.array('h')
    for i in range(1000): d.append(i % 30000); d.append((i+1) % 30000)
    w = tc.Writer('/tmp/bench_enc.des9', 2, 128)
    t = _ms()
    for _ in range(20):
        w.write_rows(d)
    elapsed = _diff(t)
    w.close()
    os.remove('/tmp/bench_enc.des9')
    rows_per_ms = 20000 / elapsed if elapsed else 0
    print("  bench_encode: 20x1000 rows in {}ms  ({:.0f} rows/ms)".format(elapsed, rows_per_ms))
    check(elapsed < 5000, "encode too slow: {}ms".format(elapsed))

def bench_decode_1000_rows():
    """Benchmark read_chunk_into: decode 1000 rows, 2 cols, repeated 20x."""
    d = array.array('h')
    for i in range(1000): d.append(i % 30000); d.append((i+1) % 30000)
    with tc.Writer('/tmp/bench_dec.des9', 2, 128) as w:
        w.write_rows(d)
    buf = array.array('h', [0] * (128 * 2))
    t = _ms()
    for _ in range(20):
        with tc.Reader('/tmp/bench_dec.des9') as r:
            while r.read_chunk_into(buf): pass
    elapsed = _diff(t)
    os.remove('/tmp/bench_dec.des9')
    rows_per_ms = 20000 / elapsed if elapsed else 0
    print("  bench_decode: 20x1000 rows in {}ms  ({:.0f} rows/ms)".format(elapsed, rows_per_ms))
    check(elapsed < 5000, "decode too slow: {}ms".format(elapsed))

def bench_struct_unpack():
    """Benchmark struct.unpack patterns: per-word vs bulk."""
    import struct
    raw = bytes(range(256)) * 16  # 4096 bytes = 1024 uint32 words
    n   = 1024

    # Per-word unpack (current approach)
    t = _ms()
    for _ in range(100):
        for wi in range(n):
            struct.unpack('<I', raw[wi*4:wi*4+4])
    t_per_word = _diff(t)

    # Bulk unpack
    fmt = '<{}I'.format(n)
    t = _ms()
    for _ in range(100):
        struct.unpack(fmt, raw[:n*4])
    t_bulk = _diff(t)

    print("  struct.unpack 1024 words x100: per-word={}ms  bulk={}ms  speedup={:.1f}x".format(
        t_per_word, t_bulk, t_per_word/t_bulk if t_bulk else 0))

def bench_decode_detail():
    """Break down decode cost per chunk."""
    import struct

    # Build a real file with 256 rows, 7 cols
    n_rows, n_cols = 256, 7
    d = array.array('h')
    for i in range(n_rows):
        for c in range(n_cols): d.append((i + c*100) % 30000)

    path = '/tmp/bench_detail.des9'
    with tc.Writer(path, n_cols, n_rows) as w:
        w.write_rows(d)

    decode_buf = array.array('h', [0] * (n_rows * n_cols))
    N = 200  # iterations

    # Full decode (open + read_chunk_into + close)
    t = _ms()
    for _ in range(N):
        r = tc.Reader(path)
        r.read_chunk_into(decode_buf)
        r.close()
    t_full = _diff(t)

    # Just file open + read raw bytes + close (no decode)
    import os
    fsize = os.stat(path)[6]
    t = _ms()
    for _ in range(N):
        f = open(path, 'rb')
        f.read(fsize)
        f.close()
    t_raw_io = _diff(t)

    # Just struct.unpack (simulated: unpack n_cols * n_words words)
    # Estimate ~20 words per col per chunk
    n_words = n_cols * 20
    raw = bytes(range(128)) * ((n_words * 4 // 128) + 1)
    fmt = '<{}I'.format(n_words)
    t = _ms()
    for _ in range(N):
        struct.unpack(fmt, raw[:n_words * 4])
    t_unpack = _diff(t)

    # Just the bit-extraction inner loop (worst case: 28 words, 1 val each)
    words = struct.unpack('<28I', bytes(range(112)))
    dst = array.array('h', [0] * (n_rows * n_cols))
    t = _ms()
    for _ in range(N):
        prev = 0; idx = 0
        for word in words:
            v = word & 0x1FFFF
            prev += (v >> 1) ^ -(v & 1)
            dst[idx] = prev; idx += 1
    t_bits = _diff(t)

    # Array slice allocation
    t = _ms()
    for _ in range(N):
        array.array('h', decode_buf[0 : n_rows * n_cols])
    t_slice = _diff(t)

    os.remove(path)
    print("  decode_detail x{} ({}rows {}cols):".format(N, n_rows, n_cols))
    print("    raw_io={}ms  full_decode={}ms  unpack={}ms  bits={}ms  slice={}ms".format(
        t_raw_io, t_full, t_unpack, t_bits, t_slice))
    print("    decode overhead vs raw IO: {:.1f}x".format(t_full / t_raw_io if t_raw_io else 0))


def bench_readinto_vs_read():
    """Compare readinto(buf,n) vs read(n) and struct.unpack_from vs unpack."""
    import struct
    N = 1000

    path = '/tmp/bench_rio.bin'
    data = bytes(range(256)) * 40  # 10240 bytes
    with open(path, 'wb') as f:
        f.write(data)

    buf6  = bytearray(6)
    buf1k = bytearray(1024)

    # readinto(buf, n) — header size
    t = _ms()
    for _ in range(N):
        f = open(path, 'rb')
        for _ in range(14):  # 7 cols * 2 reads each
            f.readinto(buf6, 6)
        f.close()
    t_readinto_hdr = _diff(t)

    # readinto(buf, n) — data size (~140 bytes typical)
    t = _ms()
    for _ in range(N):
        f = open(path, 'rb')
        for _ in range(7):
            f.readinto(buf1k, 140)
        f.close()
    t_readinto_dat = _diff(t)

    # read(n) — for comparison
    t = _ms()
    for _ in range(N):
        f = open(path, 'rb')
        for _ in range(14):
            f.read(6)
        f.close()
    t_read_hdr = _diff(t)

    # struct.unpack_from vs unpack
    t = _ms()
    fmt = '<hHH'
    for _ in range(N * 7):
        struct.unpack_from(fmt, buf6)
    t_upk_from = _diff(t)

    t = _ms()
    for _ in range(N * 7):
        struct.unpack(fmt, bytes(buf6))
    t_upk = _diff(t)

    # open+close cost alone
    t = _ms()
    for _ in range(N):
        f = open(path, 'rb')
        f.close()
    t_open = _diff(t)

    os.remove(path)
    print("  readinto_hdr(6)x14 x{}: {}ms  ({:.1f}ms/chunk)".format(N, t_readinto_hdr, t_readinto_hdr/N))
    print("  readinto_dat(140)x7 x{}: {}ms  ({:.1f}ms/chunk)".format(N, t_readinto_dat, t_readinto_dat/N))
    print("  read(6)x14         x{}: {}ms  ({:.1f}ms/chunk)".format(N, t_read_hdr, t_read_hdr/N))
    print("  unpack_from x{}:    {}ms".format(N*7, t_upk_from))
    print("  unpack      x{}:    {}ms".format(N*7, t_upk))
    print("  open+close  x{}:    {}ms  ({:.1f}ms/op)".format(N, t_open, t_open/N))

BENCH_TESTS = [
    bench_struct_unpack,
    bench_encode_1000_rows,
    bench_decode_1000_rows,
    bench_decode_detail,
    bench_readinto_vs_read,
]

print("\n--- benchmarks ---")
for b in BENCH_TESTS:
    try:
        b()
    except Exception as e:
        print("BENCH FAIL:", b.__name__, "-", e)

def bench_encode_detail():
    """Break down encode cost: zigzag vs selector search vs file write."""
    import struct
    d = array.array('h')
    for i in range(1000): d.append(i % 30000); d.append((i+1) % 30000)
    win  = array.array('I', [0] * 28)
    wbuf = array.array('I', [0] * 128)

    # Just the zigzag+delta loop (window fill only, no selector)
    t = _ms()
    for _ in range(100):
        prev = d[0]
        pending = 0
        for i in range(0, len(d), 2):  # col 0, stride 2
            dv = d[i] - prev
            prev = d[i]
            win[pending % 28] = ((dv << 1) ^ (dv >> 16)) & 0x1FFFF
            pending += 1
    t_zigzag = _diff(t)

    # Just struct.pack calls (header writes)
    t = _ms()
    hdr = struct.pack('<hHH', 0, 20, 128)
    for _ in range(100):
        for _ in range(8 * 2):  # 8 chunks * 2 cols
            struct.pack('<hHH', 0, 20, 128)
    t_pack = _diff(t)

    # Just bytes() conversion
    t = _ms()
    for _ in range(100):
        for _ in range(16):  # 8 chunks * 2 cols
            bytes(wbuf[:20])
    t_bytes = _diff(t)

    print("  encode detail x100: zigzag_loop={}ms  struct_pack={}ms  bytes_conv={}ms".format(
        t_zigzag, t_pack, t_bytes))
