"""
microhive.py - Timeseries database for MicroPython / ESP32

Partition layout (hive-style):
  date=YYYY-MM-DD/                    (granularity='date')
  date=YYYY-MM-DD/hour=HH/            (granularity='hour')
  date=YYYY-MM-DD/hour=HH/min=MM/     (granularity='minute')

Each partition contains one .npy file per resource, named <resource>.npy
Row-based (uncompressed). Compaction deferred post-prototype.

Time is dense: row index * hop_us = offset from partition start timestamp (us).
No explicit time column stored.
"""

import os
import array
import time

import npyfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPECODE = 'h'  # int16
ITEMSIZE = 2    # bytes per element

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _epoch_to_parts(epoch_s):
    """Convert Unix epoch seconds to (year, month, day, hour, minute, second)."""
    t = time.gmtime(epoch_s)
    return t[0], t[1], t[2], t[3], t[4], t[5]

def _parts_to_epoch(year, month, day, hour=0, minute=0, second=0):
    """Convert UTC calendar parts to Unix epoch seconds (MicroPython compatible)."""
    # MicroPython time.mktime expects a tuple of 8 elements
    # MicroPython: 8-tuple. CPython: 9-tuple with DST=-1. Try both.
    try:
        return int(time.mktime((year, month, day, hour, minute, second, 0, 0)))
    except TypeError:
        return int(time.mktime((year, month, day, hour, minute, second, 0, 0, -1)))

def _partition_epoch(year, month, day, hour=0, minute=0):
    """Epoch seconds for the start of a partition."""
    return _parts_to_epoch(year, month, day, hour, minute, 0)

# ---------------------------------------------------------------------------
# Partition path helpers
# ---------------------------------------------------------------------------

def _partition_dir(base_dir, resource_name, granularity, year, month, day, hour=0, minute=0):
    date_part = "date={:04d}-{:02d}-{:02d}".format(year, month, day)
    if granularity == 'date':
        return "{}/{}/{}".format(base_dir, resource_name, date_part)
    elif granularity == 'hour':
        return "{}/{}/{}/hour={:02d}".format(base_dir, resource_name, date_part, hour)
    elif granularity == 'minute':
        return "{}/{}/{}/hour={:02d}/min={:02d}".format(
            base_dir, resource_name, date_part, hour, minute)
    else:
        raise ValueError("Unknown granularity: {}".format(granularity))

def _data_file(partition_dir, resource_name):
    return "{}/{}.npy".format(partition_dir, resource_name)

def _partition_duration_s(granularity):
    """Duration of one partition in seconds."""
    if granularity == 'date':
        return 86400
    elif granularity == 'hour':
        return 3600
    elif granularity == 'minute':
        return 60
    else:
        raise ValueError("Unknown granularity: {}".format(granularity))

def _epoch_to_partition_key(epoch_s, granularity):
    """Return (year, month, day, hour, minute) tuple for a partition start."""
    y, mo, d, h, mi, _ = _epoch_to_parts(epoch_s)
    if granularity == 'date':
        return (y, mo, d, 0, 0)
    elif granularity == 'hour':
        return (y, mo, d, h, 0)
    elif granularity == 'minute':
        return (y, mo, d, h, mi)
    else:
        raise ValueError("Unknown granularity: {}".format(granularity))

def _partition_start_epoch(key):
    """(y, mo, d, h, mi) → epoch seconds."""
    y, mo, d, h, mi = key
    return _partition_epoch(y, mo, d, h, mi)

# ---------------------------------------------------------------------------
# Partition enumeration
# ---------------------------------------------------------------------------

def _list_dir_safe(path):
    """os.listdir returning [] if path does not exist."""
    try:
        return os.listdir(path)
    except OSError:
        return []

def _enumerate_partitions(base_dir, resource_name, granularity):
    """
    Yield (partition_key, partition_dir) for all existing partitions,
    in sorted order. partition_key = (year, month, day, hour, minute).
    """
    res_dir = "{}/{}".format(base_dir, resource_name)
    date_dirs = sorted(_list_dir_safe(res_dir))
    for dd in date_dirs:
        if not dd.startswith("date="):
            continue
        date_str = dd[5:]  # YYYY-MM-DD
        try:
            y, mo, d = int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10])
        except (ValueError, IndexError):
            continue
        date_path = "{}/{}".format(res_dir, dd)
        if granularity == 'date':
            yield (y, mo, d, 0, 0), date_path
        elif granularity == 'hour':
            for hd in sorted(_list_dir_safe(date_path)):
                if not hd.startswith("hour="):
                    continue
                try:
                    h = int(hd[5:])
                except ValueError:
                    continue
                yield (y, mo, d, h, 0), "{}/{}".format(date_path, hd)
        elif granularity == 'minute':
            for hd in sorted(_list_dir_safe(date_path)):
                if not hd.startswith("hour="):
                    continue
                try:
                    h = int(hd[5:])
                except ValueError:
                    continue
                hour_path = "{}/{}".format(date_path, hd)
                for md in sorted(_list_dir_safe(hour_path)):
                    if not md.startswith("min="):
                        continue
                    try:
                        mi = int(md[4:])
                    except ValueError:
                        continue
                    yield (y, mo, d, h, mi), "{}/{}".format(hour_path, md)

def _partitions_in_range(base_dir, resource_name, granularity, start_s, end_s):
    """
    Yield (partition_key, partition_dir, part_start_s, part_end_s)
    for partitions whose time range overlaps [start_s, end_s].
    Only enumerates directories that can possibly overlap.
    """
    dur = _partition_duration_s(granularity)
    for key, pdir in _enumerate_partitions(base_dir, resource_name, granularity):
        ps = _partition_start_epoch(key)
        pe = ps + dur
        if ps >= end_s:
            break  # sorted order: no more partitions can overlap
        if pe > start_s:
            yield key, pdir, ps, pe

# ---------------------------------------------------------------------------
# File row count helper
# ---------------------------------------------------------------------------

def _file_row_count(path, n_cols):
    """Return number of rows in a .npy file without reading data."""
    try:
        with npyfile.Reader(path) as r:
            total_items = r.shape[0] * r.shape[1] if len(r.shape) == 2 else r.shape[0]
            return total_items // n_cols
    except OSError:
        return 0

# ---------------------------------------------------------------------------
# Append helpers
# ---------------------------------------------------------------------------

def _makedirs(path):
    """Recursively create directories (MicroPython os.makedirs may not exist)."""
    parts = path.split('/')
    is_abs = path.startswith('/')
    current = '/' if is_abs else ''
    for part in parts:
        if not part:
            continue
        if current == '/':
            current = '/' + part
        elif current == '':
            current = part
        else:
            current = current + '/' + part
        try:
            os.mkdir(current)
        except OSError:
            pass  # already exists

def _append_rows_to_file(path, data, n_cols):
    """
    Append rows (flat array.array row-major) to an existing or new .npy file.
    .npy doesn't support in-place header updates for shape, so we:
      1. Read existing data
      2. Append new data
      3. Rewrite the file
    This is acceptable pre-compaction since files are small.
    """
    n_new_rows = len(data) // n_cols

    # Read existing data if file exists
    existing = array.array(TYPECODE)
    existing_rows = 0
    try:
        with npyfile.Reader(path) as r:
            existing_rows = r.shape[0]
            for chunk in r.read_data_chunks(n_cols):
                existing.extend(chunk)
    except OSError:
        pass  # file doesn't exist yet

    total_rows = existing_rows + n_new_rows
    combined = existing
    combined.extend(data)

    _makedirs('/'.join(path.split('/')[:-1]))
    with npyfile.Writer(path, shape=(total_rows, n_cols), typecode=TYPECODE) as w:
        w.write_values(combined, typecode=TYPECODE)

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MicroHive:
    """
    Timeseries database for MicroPython.

    resources dict example:
        {
            "raw": {
                "hop": 50000,           # microseconds between samples (20Hz = 50000us)
                "columns": ["a", "b"],
                "dtype": "int16",       # only int16 supported currently
                "granularity": "minute" # "date" | "hour" | "minute"
            }
        }
    """

    DEFAULT_CHUNK_ROWS = 64

    def __init__(self, base_dir, resources):
        self._base = base_dir
        self._resources = resources
        # Track last partition key per resource to detect boundary crossings
        self._last_partition = {}

    def _res(self, name):
        if name not in self._resources:
            raise ValueError("Unknown resource: {}".format(name))
        return self._resources[name]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_timerange(self, resource, start_s, end_s, chunk_rows=None):
        """
        Query [start_s, end_s) in Unix epoch seconds.
        Yields array.array('h') of chunk_rows rows in row-major order.
        Each element corresponds to resource['columns'] in order.
        """
        if chunk_rows is None:
            chunk_rows = self.DEFAULT_CHUNK_ROWS
        cfg = self._res(resource)
        n_cols = len(cfg['columns'])
        gran = cfg['granularity']
        hop_us = cfg['hop']
        dur_s = _partition_duration_s(gran)

        buf = array.array(TYPECODE)
        buf_rows = 0

        for key, pdir, ps, pe in _partitions_in_range(
                self._base, resource, gran, start_s, end_s):

            path = _data_file(pdir, resource)
            print('open', path)
            try:
                reader = npyfile.Reader(path)
            except OSError:
                continue

            with reader as r:
                if len(r.shape) < 2:
                    continue
                total_rows = r.shape[0]
                file_cols = r.shape[1]
                if file_cols != n_cols:
                    continue

                hop_s_float = hop_us / 1_000_000.0
                # Row index range that overlaps [start_s, end_s)
                # row_time = ps + row_idx * hop_s
                row_start = max(0, int((start_s - ps) / hop_s_float))
                if end_s >= pe:
                    row_end = total_rows
                else:
                    row_end = min(total_rows, int((end_s - ps) / hop_s_float) + 1)

                if row_start >= row_end:
                    continue

                row_idx = 0
                items_to_skip = row_start * n_cols
                items_skipped = 0

                for chunk in r.read_data_chunks(n_cols):
                    # Skip rows before row_start
                    if items_skipped < items_to_skip:
                        items_skipped += len(chunk)
                        continue

                    if row_idx >= row_end:
                        break

                    if row_start <= row_idx < row_end:
                        buf.extend(chunk)
                        buf_rows += 1
                        if buf_rows >= chunk_rows:
                            yield buf
                            buf = array.array(TYPECODE)
                            buf_rows = 0

                    row_idx += 1

        if buf_rows > 0:
            yield buf

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_data(self, resource, data, timestamp_s=None):
        """
        Append rows to the appropriate time partition.
        data: flat array.array('h') in row-major order.
        Length must be a multiple of len(columns).
        timestamp_s: Unix epoch seconds for the first sample. Defaults to time.time().
                     Pass explicitly when loading historical data.
        Raises ValueError on bad input or out-of-order writes.
        Auto-splits across partition boundaries.
        """
        cfg = self._res(resource)
        n_cols = len(cfg['columns'])
        gran = cfg['granularity']
        hop_us = cfg['hop']

        if len(data) == 0:
            raise ValueError("data is empty")
        if len(data) % n_cols != 0:
            raise ValueError(
                "data length {} not a multiple of column count {}".format(len(data), n_cols))

        n_rows = len(data) // n_cols
        now_s = timestamp_s if timestamp_s is not None else time.time()
        dur_s = _partition_duration_s(gran)
        hop_s_float = hop_us / 1_000_000.0

        # Determine which partition this timestamp falls into
        current_key = _epoch_to_partition_key(now_s, gran)
        part_start_s = _partition_start_epoch(current_key)
        part_end_s = part_start_s + dur_s

        # Check for out-of-order: if last partition for this resource is ahead of now
        last_key = self._last_partition.get(resource)
        if last_key is not None and last_key > current_key:
            raise ValueError(
                "Out-of-order write: last partition {} > current {}".format(last_key, current_key))

        self._last_partition[resource] = current_key

        # Check if data spans a partition boundary
        last_sample_s = now_s + (n_rows - 1) * hop_s_float

        if last_sample_s < part_end_s:
            # All data fits in current partition
            pdir = _partition_dir(self._base, resource, gran,
                                  current_key[0], current_key[1], current_key[2],
                                  current_key[3], current_key[4])
            _makedirs(pdir)
            path = _data_file(pdir, resource)
            _append_rows_to_file(path, data, n_cols)
        else:
            # Split: write rows that fit in current partition, recurse for the rest
            remaining_s = part_end_s - now_s
            rows_in_current = max(0, int(remaining_s / hop_s_float))

            if rows_in_current > 0:
                chunk1 = array.array(TYPECODE, data[:rows_in_current * n_cols])
                pdir = _partition_dir(self._base, resource, gran,
                                      current_key[0], current_key[1], current_key[2],
                                      current_key[3], current_key[4])
                _makedirs(pdir)
                path = _data_file(pdir, resource)
                _append_rows_to_file(path, chunk1, n_cols)

            remaining_data = array.array(TYPECODE, data[rows_in_current * n_cols:])
            if len(remaining_data) > 0:
                self.append_data(resource, remaining_data,
                                 timestamp_s=part_end_s)

    # ------------------------------------------------------------------
    # Info / debug
    # ------------------------------------------------------------------

    def get_info(self, resource):
        """Return dict with resource config and basic stats."""
        cfg = self._res(resource)
        gran = cfg['granularity']
        n_partitions = sum(1 for _ in _enumerate_partitions(
            self._base, resource, gran))
        return {
            'resource': resource,
            'granularity': gran,
            'hop_us': cfg['hop'],
            'columns': cfg['columns'],
            'n_partitions': n_partitions,
        }
