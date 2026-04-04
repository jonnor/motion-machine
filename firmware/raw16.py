"""
raw16.py - Raw int16 timeseries storage for MicroPython.

Same Writer/Reader interface as delta_simple9 but stores int16 values
uncompressed. No encoding overhead — read speed is limited only by flash IO.

File format:
  Header: magic(4) + version(1) + n_cols(1) + chunk_size(2) = 8 bytes
  Chunks: n_rows(uint16) + data(n_rows * n_cols * int16)
"""

import array
import struct

_MAGIC        = b'RW16'
_VERSION      = 1
_HDR_FMT      = '<4sBBH'
_HDR_SIZE     = struct.calcsize(_HDR_FMT)   # 8 bytes
_CHUNK_HDR_FMT  = '<H'
_CHUNK_HDR_SIZE = 2  # uint16 n_rows

# ---------------------------------------------------------------------------
class Writer:
    """
    Writes raw int16 data in chunk_size-row blocks.
    Appends to existing file if present and compatible.
    Use as context manager or call close() when done.
    """

    def __init__(self, path, n_cols, chunk_size=128):
        self._n_cols     = n_cols
        self._chunk_size = chunk_size
        self._buf_rows   = 0
        self._buf        = array.array('h', [0] * (n_cols * chunk_size))

        try:
            with open(path, 'rb') as f:
                magic, ver, nc, cs = struct.unpack(_HDR_FMT, f.read(_HDR_SIZE))
            if magic != _MAGIC:
                raise ValueError("Not a raw16 file")
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

        fw  = self._file.write
        buf = self._buf
        pos = offset

        # Flush any buffered partial chunk first
        if self._buf_rows > 0:
            self._flush()

        # Write full chunks directly from data via memoryview — no copy
        mv = memoryview(data)
        while n_rows >= cs:
            fw(struct.pack(_CHUNK_HDR_FMT, cs))
            fw(mv[pos:pos + cs * nc])
            pos    += cs * nc
            n_rows -= cs

        # Buffer remaining partial rows
        if n_rows > 0:
            for i in range(n_rows * nc):
                buf[i] = data[pos + i]
            self._buf_rows = n_rows

    def write_row(self, row, offset=0):
        """Append one row from row[offset:offset+n_cols]."""
        buf_r = self._buf_rows
        buf   = self._buf
        nc    = self._n_cols
        for c in range(nc):
            buf[buf_r * nc + c] = row[offset + c]
        self._buf_rows = buf_r + 1
        if self._buf_rows >= self._chunk_size:
            self._flush()

    def flush(self):
        """Flush buffered rows and fsync to disk."""
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

    def _flush(self):
        n  = self._buf_rows
        nc = self._n_cols
        self._file.write(struct.pack(_CHUNK_HDR_FMT, n))
        self._file.write(bytes(array.array('h', self._buf[:n * nc])))
        self._buf_rows = 0


# ---------------------------------------------------------------------------
class Reader:
    """
    Reads raw int16 data written by Writer.
    Use read_chunk_into() in a loop until it returns 0.
    """

    def __init__(self, path):
        self._file = open(path, 'rb')
        hdr = self._file.read(_HDR_SIZE)
        magic, ver, nc, cs = struct.unpack(_HDR_FMT, hdr)
        if magic != _MAGIC:
            raise ValueError("Not a raw16 file")
        self._n_cols     = nc
        self._chunk_size = cs
        # Pre-allocated read buffer: worst-case full chunk
        self._hdr_buf = bytearray(_CHUNK_HDR_SIZE)
        self._raw_buf = bytearray(cs * nc * 2)

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
        Reads directly into out with readinto — zero allocation, zero copy.
        """
        hdr = self._hdr_buf
        if not self._file.readinto(hdr, _CHUNK_HDR_SIZE):
            return 0
        n_rows = struct.unpack(_CHUNK_HDR_FMT, hdr)[0]
        # readinto array.array('h') directly — MicroPython reads raw bytes
        self._file.readinto(memoryview(out)[:n_rows * self._n_cols])
        return n_rows

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
