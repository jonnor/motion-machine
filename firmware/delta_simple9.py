"""
delta_simple9.py - Timeseries compression library for MicroPython
Supports: int16, multiple columns, Simple9 encoding, delta+zigzag transforms.

Chunk format per column:
  [seed: int16 LE][n_words: uint16 LE][n_rows: uint16 LE][words: n_words * uint32 LE]

  seed = first value of the chunk (delta decode starting point).
  Values are delta-encoded then 17-bit zigzag-encoded before Simple9 compression.
"""

import array
import struct

# ---------------------------------------------------------------------------
# Simple9 selector table: (selector_index, bits_per_int, ints_per_word)
# ---------------------------------------------------------------------------
_S9_SELECTORS = (
    (0,  1, 28),
    (1,  2, 14),
    (2,  3,  9),
    (3,  4,  7),
    (4,  5,  5),
    (5,  7,  4),
    (6,  9,  3),
    (7, 14,  2),
    (8, 28,  1),
)
_S9_MAX_WINDOW = 28

# ---------------------------------------------------------------------------
# Simple9 encoder
#
# Reads int16 values from src[base:base+n], applies delta+zigzag inline,
# and writes encoded uint32 words into dst[0:n_words].
# win is a pre-allocated array.array('I') of length >= _S9_MAX_WINDOW.
# Returns the number of words written.
# ---------------------------------------------------------------------------

def _s9_encode(src, base, n, dst, win, stride=1):
    """
    Delta+zigzag-encode n values from src starting at base with given stride,
    Simple9-compressing into pre-allocated array.array('I') dst.
    stride=1: contiguous column-major (default, from _buf).
    stride=n_cols: row-major column access (direct from input data).
    Returns number of words written.
    The first value is used as the delta seed (first delta is always 0).
    """
    src_i   = base
    src_end = base + n * stride
    n_words = 0
    prev    = src[base]
    exhausted = False
    pending   = 0

    while not exhausted or pending > 0:
        while pending < _S9_MAX_WINDOW:
            if src_i >= src_end:
                exhausted = True
                break
            d = src[src_i] - prev
            prev         = src[src_i]
            src_i       += stride
            win[pending] = ((d << 1) ^ (d >> 16)) & 0x1FFFF
            pending     += 1

        if pending == 0:
            break

        for sel, bits, count in _S9_SELECTORS:
            if count > pending and not exhausted:
                continue
            take    = min(count, pending)
            max_val = (1 << bits) - 1
            ok = True
            for j in range(take):
                if win[j] > max_val:
                    ok = False
                    break
            if ok:
                word = sel << 28
                for j in range(take):
                    word |= win[j] << (j * bits)
                dst[n_words] = word & 0xFFFFFFFF
                n_words += 1
                rem = pending - take
                for j in range(rem):
                    win[j] = win[j + take]
                pending = rem
                break
        else:
            raise ValueError("Value too large for Simple9")

    return n_words


# ---------------------------------------------------------------------------
# Simple9 decoder
#
# Decodes n_words words from raw bytes, undoing zigzag+delta inline, and
# writes n_values results into dst[base], dst[base+stride], ...
# seed is the first value of the chunk (delta starting point).
# ---------------------------------------------------------------------------

def _s9_decode_into(raw, n_words, n_values, dst, base, stride, seed):
    """
    Decode n_words Simple9 words from raw bytes, undoing zigzag+delta inline.
    Writes results into dst starting at base with given stride.
    Stops after n_values values.
    """
    emitted = 0
    idx     = base
    prev    = seed
    for wi in range(n_words):
        word = struct.unpack('<I', raw[wi*4:wi*4+4])[0]
        sel  = (word >> 28) & 0xF
        if sel >= len(_S9_SELECTORS):
            raise ValueError("Invalid Simple9 selector: {}".format(sel))
        _, bits, count = _S9_SELECTORS[sel]
        mask = (1 << bits) - 1
        take = min(count, n_values - emitted)
        for j in range(take):
            v     = (word >> (j * bits)) & mask
            prev += (v >> 1) ^ -(v & 1)
            dst[idx] = prev
            idx     += stride
            emitted += 1
        if emitted >= n_values:
            break


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------
# File header: magic(4) + version(1) + n_cols(1) + chunk_size(2) = 8 bytes
_FILE_MAGIC = b'DES9'
_FILE_VERSION = 1
_FILE_HEADER_FMT = '<4sBBH'
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_FMT)  # 8 bytes

# Chunk column header: seed(int16) + n_words(uint16) + n_rows(uint16) = 6 bytes
_COL_HEADER_FMT = '<hHH'
_COL_HEADER_SIZE = struct.calcsize(_COL_HEADER_FMT)  # 6 bytes


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class Writer:
    """
    Append-capable writer for delta+Simple9 compressed timeseries files.

    Parameters
    ----------
    path       : file path
    n_cols     : number of int16 columns
    chunk_size : rows per chunk (default 128)
    """

    def __init__(self, path, n_cols, chunk_size=128):
        self._n_cols     = n_cols
        self._chunk_size = chunk_size
        self._buf_rows   = 0
        # Flat buffer: column c at [c*chunk_size : (c+1)*chunk_size].
        self._buf      = array.array('h', [0] * (n_cols * chunk_size))
        # Worst-case Simple9 output: one word per input value.
        self._word_buf = array.array('I', [0] * chunk_size)
        # Scratch window for encoder, shared across all _flush calls.
        self._win      = array.array('I', [0] * _S9_MAX_WINDOW)

        try:
            with open(path, 'rb') as f:
                magic, ver, nc, cs = struct.unpack(_FILE_HEADER_FMT,
                                                   f.read(_FILE_HEADER_SIZE))
            if magic != _FILE_MAGIC:
                raise ValueError("Not a delta_simple9 file")
            if nc != n_cols or cs != chunk_size:
                raise ValueError("File header mismatch")
            self._file = open(path, 'ab')
        except OSError:
            self._file = open(path, 'wb')
            self._file.write(struct.pack(_FILE_HEADER_FMT,
                                         _FILE_MAGIC, _FILE_VERSION,
                                         n_cols, chunk_size))
            self._file.flush()

    def write_rows(self, data, offset=0, n_rows=None):
        """Append rows from array.array('h') in row-major C layout.
        offset: starting item index (not row index) into data.
        n_rows: number of rows to write; defaults to all rows from offset.
        Encodes directly from row-major data in chunk_size batches via stride.
        Any remainder is encoded immediately as a partial chunk — no row-by-row
        Python loops, no deferred buffering across calls."""
        nc  = self._n_cols
        cs  = self._chunk_size
        if n_rows is None:
            n_rows = (len(data) - offset) // nc

        fw       = self._file.write
        word_buf = self._word_buf
        win      = self._win
        pos      = offset

        # Flush any previously buffered partial chunk first
        if self._buf_rows > 0:
            self._flush()

        # Encode all data in chunk_size batches directly from row-major input
        while n_rows > 0:
            take = min(n_rows, cs)
            for c in range(nc):
                seed    = data[pos + c]
                n_words = _s9_encode(data, pos + c, take, word_buf, win, nc)
                fw(struct.pack(_COL_HEADER_FMT, seed, n_words, take))
                fw(bytes(word_buf[:n_words]))
            pos    += take * nc
            n_rows -= take

        self._file.flush()

    def write_row(self, row, offset=0):
        """Append one row: values taken from row[offset:offset+n_cols]."""
        buf_r = self._buf_rows
        buf   = self._buf
        cs    = self._chunk_size
        for c in range(self._n_cols):
            buf[c * cs + buf_r] = row[offset + c]
        self._buf_rows = buf_r + 1
        if self._buf_rows >= cs:
            self._flush()

    def flush(self):
        """Force flush any buffered rows as a partial chunk."""
        if self._buf_rows > 0:
            self._flush()

    def close(self):
        self.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _flush(self):
        fw       = self._file.write
        buf      = self._buf
        word_buf = self._word_buf
        win      = self._win
        cs       = self._chunk_size
        n_rows   = self._buf_rows

        for c in range(self._n_cols):
            base    = c * cs
            seed    = buf[base]
            n_words = _s9_encode(buf, base, n_rows, word_buf, win)
            fw(struct.pack(_COL_HEADER_FMT, seed, n_words, n_rows))
            fw(bytes(word_buf[:n_words]))

        self._file.flush()
        self._buf_rows = 0


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class Reader:
    """
    Sequential reader for delta+Simple9 compressed timeseries files.

    Primary interface: read_chunk_into(out) — decodes one chunk directly into
    a pre-allocated array.array('h') in row-major C layout.
    Returns the number of rows written, or 0 at EOF.

    read_chunk() allocates and calls read_chunk_into().
    read_all() accumulates all chunks into one array.
    """

    def __init__(self, path):
        self._file = open(path, 'rb')
        hdr = self._file.read(_FILE_HEADER_SIZE)
        magic, ver, nc, cs = struct.unpack(_FILE_HEADER_FMT, hdr)
        if magic != _FILE_MAGIC:
            raise ValueError("Not a delta_simple9 file")
        self._n_cols     = nc
        self._chunk_size = cs
        self._data_start = _FILE_HEADER_SIZE

    @property
    def n_cols(self):
        return self._n_cols

    @property
    def chunk_size(self):
        return self._chunk_size

    def read_chunk_into(self, out):
        """
        Decode one chunk directly into pre-allocated array.array('h') out.
        out must have capacity >= chunk_size * n_cols, laid out row-major.
        Column c of row r is written to out[r*n_cols + c].
        Returns number of rows written, or 0 at EOF.
        """
        hdr_bytes = self._file.read(_COL_HEADER_SIZE)
        if not hdr_bytes or len(hdr_bytes) < _COL_HEADER_SIZE:
            return 0

        seed0, n_words0, n_rows = struct.unpack(_COL_HEADER_FMT, hdr_bytes)
        nc = self._n_cols

        for c in range(nc):
            if c == 0:
                seed, nw = seed0, n_words0
            else:
                hb = self._file.read(_COL_HEADER_SIZE)
                seed, nw, _ = struct.unpack(_COL_HEADER_FMT, hb)

            raw = self._file.read(nw * 4)
            _s9_decode_into(raw, nw, n_rows, out, c, nc, seed)

        return n_rows

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
