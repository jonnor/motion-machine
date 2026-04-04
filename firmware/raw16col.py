"""
raw16col.py - Column-wise raw int16 timeseries storage for MicroPython.

Same Writer/Reader interface as delta_simple9/raw16 but stores data
column-major: each chunk stores all values of col0, then col1, etc.

File format:
  Header: magic(4) + version(1) + n_cols(1) + chunk_size(2) = 8 bytes
  Chunks: n_rows(uint16) + col0[n_rows * int16] + col1[n_rows * int16] + ...

For all-column reads this gives the same IO as raw16 but with simpler
readinto calls. The key benefit is that partial-column reads are possible
with seeking.
"""

import array
import struct

_MAGIC          = b'RC16'
_VERSION        = 1
_HDR_FMT        = '<4sBBH'
_HDR_SIZE       = struct.calcsize(_HDR_FMT)   # 8 bytes
_CHUNK_HDR_FMT  = '<H'
_CHUNK_HDR_SIZE = 2

# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, path, n_cols, chunk_size=128):
        self._n_cols     = n_cols
        self._chunk_size = chunk_size
        self._buf_rows   = 0
        # Row-major input buffer (same as raw16/delta_simple9)
        self._buf = array.array('h', [0] * (n_cols * chunk_size))
        # Column-major write buffer (bytearray for direct file write)
        self._col_buf = bytearray(chunk_size * 2)

        try:
            with open(path, 'rb') as f:
                magic, ver, nc, cs = struct.unpack(_HDR_FMT, f.read(_HDR_SIZE))
            if magic != _MAGIC:
                raise ValueError("Not a raw16col file")
            if nc != n_cols or cs != chunk_size:
                raise ValueError("File header mismatch")
            self._file = open(path, 'ab')
        except OSError:
            self._file = open(path, 'wb')
            self._file.write(struct.pack(_HDR_FMT,
                                         _MAGIC, _VERSION, n_cols, chunk_size))

    def write_rows(self, data, offset=0, n_rows=None):
        """Append rows from array.array('h') in row-major layout."""
        nc = self._n_cols
        cs = self._chunk_size
        if n_rows is None:
            n_rows = (len(data) - offset) // nc

        if self._buf_rows > 0:
            self._flush()

        fw  = self._file.write
        pos = offset
        mv  = memoryview(data)

        while n_rows >= cs:
            fw(struct.pack(_CHUNK_HDR_FMT, cs))
            self._write_col_major(mv, pos, cs)
            pos    += cs * nc
            n_rows -= cs

        if n_rows > 0:
            buf = self._buf
            for i in range(n_rows * nc):
                buf[i] = data[pos + i]
            self._buf_rows = n_rows

    def write_row(self, row, offset=0):
        buf_r = self._buf_rows
        nc    = self._n_cols
        buf   = self._buf
        for c in range(nc):
            buf[buf_r * nc + c] = row[offset + c]
        self._buf_rows = buf_r + 1
        if self._buf_rows >= self._chunk_size:
            self._flush()

    def flush(self):
        if self._buf_rows > 0:
            self._flush()
        self._file.flush()

    def close(self):
        self.flush()
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _write_col_major(self, mv, pos, n_rows):
        """Write n_rows from row-major mv[pos:] in column-major order."""
        nc      = self._n_cols
        col_buf = self._col_buf
        fw      = self._file.write
        pack_into = struct.pack_into
        fmt = '<h'
        for c in range(nc):
            for r in range(n_rows):
                pack_into(fmt, col_buf, r * 2, mv[pos + r * nc + c])
            fw(col_buf[:n_rows * 2])

    def _flush(self):
        n  = self._buf_rows
        nc = self._n_cols
        self._file.write(struct.pack(_CHUNK_HDR_FMT, n))
        self._write_col_major(memoryview(self._buf), 0, n)
        self._buf_rows = 0


# ---------------------------------------------------------------------------
class Reader:
    def __init__(self, path):
        self._file = open(path, 'rb')
        hdr = self._file.read(_HDR_SIZE)
        magic, ver, nc, cs = struct.unpack(_HDR_FMT, hdr)
        if magic != _MAGIC:
            raise ValueError("Not a raw16col file")
        self._n_cols     = nc
        self._chunk_size = cs
        self._hdr_buf    = bytearray(_CHUNK_HDR_SIZE)
        # Per-column read buffer
        self._col_buf    = bytearray(cs * 2)

    @property
    def n_cols(self):
        return self._n_cols

    @property
    def chunk_size(self):
        return self._chunk_size

    def read_chunk_into(self, out):
        """
        Read one chunk into pre-allocated array.array('h') out (row-major).
        Returns number of rows written, or 0 at EOF.
        Reads each column with one readinto, scatters into row-major out.
        """
        hdr = self._hdr_buf
        if not self._file.readinto(hdr, _CHUNK_HDR_SIZE):
            return 0
        n_rows  = struct.unpack(_CHUNK_HDR_FMT, hdr)[0]
        nc      = self._n_cols
        col_buf = self._col_buf
        mv      = memoryview(col_buf)

        fmt = '<{}h'.format(n_rows)
        for c in range(nc):
            self._file.readinto(mv[:n_rows * 2], n_rows * 2)
            # Unpack whole column at once, scatter into row-major out
            vals = struct.unpack(fmt, col_buf[:n_rows * 2])
            for r in range(n_rows):
                out[r * nc + c] = vals[r]

        return n_rows

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
