"""
microhive.py - Timeseries database for MicroPython / ESP32

Partition layout (hive-style):
  date=YYYY-MM-DD/                    (granularity='date')
  date=YYYY-MM-DD/hour=HH/            (granularity='hour')
  date=YYYY-MM-DD/hour=HH/min=MM/     (granularity='minute')

Each partition contains one .des9 file per resource, named:
  <resource>_<first_row>.des9
where first_row is the row offset from partition start of the first stored sample.

Time is dense: (first_row + row_index) * hop_us = offset from partition start.
No explicit time column stored.
"""

import os
import array
import time

import delta_simple9

# ---------------------------------------------------------------------------
TYPECODE       = 'h'
ITEMSIZE       = 2
DS9_CHUNK_SIZE = 128

# ---------------------------------------------------------------------------
# Monotonic clock
# ---------------------------------------------------------------------------
try:
    from time import ticks_ms as _ticks_ms, ticks_diff as _ticks_diff
except ImportError:
    import time as _time
    def _ticks_ms(): return int(_time.time() * 1000)
    def _ticks_diff(new, old): return new - old

# ---------------------------------------------------------------------------
# Time / epoch helpers
# ---------------------------------------------------------------------------
# Detect epoch: MicroPython on embedded targets uses 2000-01-01 as epoch,
# but MicroPython on Unix/desktop uses 1970-01-01 (same as CPython).
# Detect by checking what gmtime(0) returns — no assumptions about platform name.
try:
    _EPOCH_OFFSET = 946684800 if time.gmtime(0)[0] == 2000 else 0
except Exception:
    _EPOCH_OFFSET = 0

def _epoch_to_parts(epoch_s):
    """Convert platform epoch seconds to UTC calendar parts via gmtime.
    gmtime expects platform-native epoch seconds (no offset needed)."""
    t = time.gmtime(int(epoch_s))
    return t[0], t[1], t[2], t[3], t[4], t[5]

# Days per month (index 0 unused)
_MDAYS = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

def _parts_to_epoch(year, month, day, hour=0, minute=0, second=0):
    """Convert UTC calendar parts to platform epoch seconds.
    Pure arithmetic — never calls time.mktime, so timezone-independent."""
    # Leap year check
    leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    # Days since Unix epoch (1970-01-01)
    y = year - 1970
    days = y * 365 + (y + 1) // 4 - (y + 69) // 100 + (y + 369) // 400
    for m in range(1, month):
        days += _MDAYS[m] + (1 if m == 2 and leap else 0)
    days += day - 1
    return int(days * 86400 + hour * 3600 + minute * 60 + second) - _EPOCH_OFFSET

def _partition_epoch(year, month, day, hour=0, minute=0):
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

def _data_file(partition_dir, resource_name, first_row):
    return "{}/{}_{}.des9".format(partition_dir, resource_name, first_row)

def _find_data_files(partition_dir, resource_name):
    """Return sorted list of (first_row, path) for all .des9 files in partition_dir."""
    prefix = resource_name + '_'
    suffix = '.des9'
    result = []
    try:
        entries = os.listdir(partition_dir)
    except OSError:
        return result
    for e in entries:
        if e.startswith(prefix) and e.endswith(suffix):
            try:
                first_row = int(e[len(prefix):-len(suffix)])
                result.append((first_row, "{}/{}".format(partition_dir, e)))
            except ValueError:
                pass
    result.sort()
    return result

def _partition_duration_s(granularity):
    if granularity == 'date':     return 86400
    elif granularity == 'hour':   return 3600
    elif granularity == 'minute': return 60
    else: raise ValueError("Unknown granularity: {}".format(granularity))

def _epoch_to_partition_key(epoch_s, granularity):
    y, mo, d, h, mi, _ = _epoch_to_parts(epoch_s)
    if granularity == 'date':     return (y, mo, d, 0, 0)
    elif granularity == 'hour':   return (y, mo, d, h, 0)
    elif granularity == 'minute': return (y, mo, d, h, mi)
    else: raise ValueError("Unknown granularity: {}".format(granularity))

def _partition_start_epoch(key):
    y, mo, d, h, mi = key
    return _partition_epoch(y, mo, d, h, mi)

# ---------------------------------------------------------------------------
# Partition enumeration
# ---------------------------------------------------------------------------
def _list_dir_safe(path):
    try:    return os.listdir(path)
    except OSError: return []

def _enumerate_partitions(base_dir, resource_name, granularity):
    res_dir   = "{}/{}".format(base_dir, resource_name)
    date_dirs = sorted(_list_dir_safe(res_dir))
    for dd in date_dirs:
        if not dd.startswith("date="): continue
        date_str = dd[5:]
        try:
            y, mo, d = int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10])
        except (ValueError, IndexError):
            continue
        date_path = "{}/{}".format(res_dir, dd)
        if granularity == 'date':
            yield (y, mo, d, 0, 0), date_path
        elif granularity == 'hour':
            for hd in sorted(_list_dir_safe(date_path)):
                if not hd.startswith("hour="): continue
                try: h = int(hd[5:])
                except ValueError: continue
                yield (y, mo, d, h, 0), "{}/{}".format(date_path, hd)
        elif granularity == 'minute':
            for hd in sorted(_list_dir_safe(date_path)):
                if not hd.startswith("hour="): continue
                try: h = int(hd[5:])
                except ValueError: continue
                hour_path = "{}/{}".format(date_path, hd)
                for md in sorted(_list_dir_safe(hour_path)):
                    if not md.startswith("min="): continue
                    try: mi = int(md[4:])
                    except ValueError: continue
                    yield (y, mo, d, h, mi), "{}/{}".format(hour_path, md)

def _partitions_in_range(base_dir, resource_name, granularity, start_s, end_s):
    dur = _partition_duration_s(granularity)
    ps  = (start_s // dur) * dur
    while ps < end_s:
        pe   = ps + dur
        key  = _epoch_to_partition_key(ps, granularity)
        pdir = _partition_dir(base_dir, resource_name, granularity,
                              key[0], key[1], key[2], key[3], key[4])
        yield key, pdir, ps, pe
        ps = pe

# ---------------------------------------------------------------------------
def _makedirs(path):
    parts   = path.split('/')
    is_abs  = path.startswith('/')
    current = '/' if is_abs else ''
    for part in parts:
        if not part: continue
        if current == '/':   current = '/' + part
        elif current == '':  current = part
        else:                current = current + '/' + part
        try:
            os.mkdir(current)
        except OSError:
            pass  # already exists
        except Exception as e:
            raise OSError("makedirs failed at {}: {}".format(current, e))

# ---------------------------------------------------------------------------
# Segment reader — reads one segment fully into out_buf, file closed before
# any yield. Returns (rows_written, resume_abs_row): resume_abs_row > 0 means
# out_buf filled up mid-segment; caller should call again with
# part_row_start=resume_abs_row and start_buf_rows=0 after yielding out_buf.
# ---------------------------------------------------------------------------
def _read_segment_into(path, n_cols, part_row_start, part_row_end, first_row,
                       decode_buf, out_buf, start_buf_rows, cap):
    """
    Read rows [part_row_start, part_row_end) from segment into out_buf
    starting at start_buf_rows, writing at most cap rows.
    File is fully closed before returning.
    Returns (rows_written, resume_abs_row): resume_abs_row==0 means fully done.
    """
    buf_rows = start_buf_rows
    file_row = first_row
    resume   = 0

    r = delta_simple9.Reader(path)
    try:
        while True:
            n = r.read_chunk_into(decode_buf)
            if n == 0:
                break

            chunk_abs_start = file_row
            chunk_abs_end   = file_row + n
            file_row        = chunk_abs_end

            if chunk_abs_end <= part_row_start:
                continue
            if chunk_abs_start >= part_row_end:
                break

            local_start = max(0, part_row_start - chunk_abs_start)
            local_end   = min(n,  part_row_end   - chunk_abs_start)

            for row_i in range(local_start, local_end):
                if buf_rows - start_buf_rows >= cap:
                    resume = chunk_abs_start + row_i
                    return buf_rows - start_buf_rows, resume
                src = row_i   * n_cols
                dst = buf_rows * n_cols
                for i in range(n_cols):
                    out_buf[dst + i] = decode_buf[src + i]
                buf_rows += 1
    finally:
        r.close()

    return buf_rows - start_buf_rows, 0

# ---------------------------------------------------------------------------
class MicroHive:
    """
    Timeseries database for MicroPython.

    resources dict example:
        {
            "sensor": {
                "hop": 1000000,         # microseconds between samples
                "columns": ["a", "b"],
                "dtype": "int16",
                "granularity": "hour"   # "date" | "hour" | "minute"
            }
        }
    """
    DEFAULT_CHUNK_ROWS = 64

    def __init__(self, base_dir, resources, debug=True):
        self._base      = base_dir
        self._resources = resources
        self._last_partition = {}
        self.debug      = debug

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
        Yields array.array('h') of up to chunk_rows rows in row-major order.
        Each segment file is fully read and closed before any yield, avoiding
        open file handles across yield points (MicroPython GC safety).
        """
        if chunk_rows is None:
            chunk_rows = self.DEFAULT_CHUNK_ROWS
        cfg    = self._res(resource)
        n_cols = len(cfg['columns'])
        gran   = cfg['granularity']
        hop_us = cfg['hop']
        # API takes Unix epoch seconds; convert to platform epoch.
        start_s = int(start_s) - _EPOCH_OFFSET
        end_s   = int(end_s)   - _EPOCH_OFFSET

        t_enum_ms = t_open_ms = t_read_ms = t_skip_ms = t_copy_ms = t_yield_ms = 0
        n_partitions = n_rows_skipped = n_rows_read = 0
        t0 = _ticks_ms()

        # Output buffer: exactly chunk_rows deep. _read_segment fills it up to
        # chunk_rows rows at a time; we yield when full, then refill.
        buf      = array.array(TYPECODE, [0] * (chunk_rows * n_cols))
        buf_rows = 0

        # Decode buffer: reused across all segment reads (one ds9 chunk at a time).
        decode_buf = array.array(TYPECODE, [0] * (DS9_CHUNK_SIZE * n_cols))

        hop_s_float = hop_us / 1_000_000.0

        t_e = _ticks_ms()
        partitions = list(_partitions_in_range(self._base, resource, gran, start_s, end_s))
        t_enum_ms += _ticks_diff(_ticks_ms(), t_e)

        for key, pdir, ps, pe in partitions:
            n_partitions += 1

            part_row_start = max(0, int((start_s - ps) / hop_s_float))
            part_row_end   = int((end_s - ps) / hop_s_float)

            t_e2 = _ticks_ms()
            files = _find_data_files(pdir, resource)
            t_enum_ms += _ticks_diff(_ticks_ms(), t_e2)

            for seg_idx, (first_row, path) in enumerate(files):
                # Skip segment entirely if it ends before our range
                if seg_idx + 1 < len(files):
                    seg_end_row = files[seg_idx + 1][0]
                    if seg_end_row <= part_row_start:
                        n_rows_skipped += seg_end_row - first_row
                        continue
                if first_row >= part_row_end:
                    break

                # Stream rows from segment directly into buf.
                # _iter_segment holds the file open across yields — safe because
                # the generator frame keeps Reader referenced (MicroPython GC tested).
                # Read segment in cap-sized slices, closing the file between
                # each yield so no Reader is open across a yield point.
                seg_start = part_row_start
                while True:
                    cap = chunk_rows - buf_rows
                    t_r = _ticks_ms()
                    written, resume = _read_segment_into(path, n_cols,
                                                         seg_start, part_row_end,
                                                         first_row, decode_buf,
                                                         buf, buf_rows, cap)
                    t_read_ms   += _ticks_diff(_ticks_ms(), t_r)
                    buf_rows    += written
                    n_rows_read += written

                    if buf_rows >= chunk_rows:
                        t_before    = _ticks_ms()
                        yield buf
                        t_yield_ms += _ticks_diff(_ticks_ms(), t_before)
                        buf_rows    = 0

                    if resume == 0:
                        break
                    seg_start = resume

        if buf_rows > 0:
            t_before = _ticks_ms()
            yield array.array(TYPECODE, buf[:buf_rows * n_cols])
            t_yield_ms += _ticks_diff(_ticks_ms(), t_before)
            buf_rows = 0

        if self.debug:
            t_total_ms = _ticks_diff(_ticks_ms(), t0)
            t_other_ms = max(0, t_total_ms - t_enum_ms - t_open_ms
                             - t_skip_ms - t_read_ms - t_copy_ms - t_yield_ms)
            print("[query debug] resource={} partitions={} rows_read={} rows_skipped={}".format(
                resource, n_partitions, n_rows_read, n_rows_skipped))
            print("[query debug] total={}ms  enum={}ms  open={}ms  skip={}ms  read={}ms  copy={}ms  yield={}ms  other={}ms".format(
                t_total_ms, t_enum_ms, t_open_ms, t_skip_ms,
                t_read_ms, t_copy_ms, t_yield_ms, t_other_ms))

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------
    def append_data(self, resource, data, timestamp_s=None):
        """
        Append rows to the appropriate time partition.
        data: flat array.array('h') in row-major order.
        timestamp_s: Unix epoch seconds for the first sample.
        Auto-splits across partition boundaries using an iterative loop.
        """
        cfg    = self._res(resource)
        n_cols = len(cfg['columns'])
        gran   = cfg['granularity']
        hop_us = cfg['hop']

        if len(data) == 0:
            raise ValueError("data is empty")
        if len(data) % n_cols != 0:
            raise ValueError(
                "data length {} not a multiple of column count {}".format(len(data), n_cols))

        # API takes Unix epoch seconds; convert to platform epoch for internal use.
        # time.time() already returns platform epoch so no conversion needed there.
        if timestamp_s is not None:
            start_s = int(timestamp_s) - _EPOCH_OFFSET
        else:
            start_s = int(time.time())
        dur_s       = _partition_duration_s(gran)
        hop_s_float = hop_us / 1_000_000.0
        n_rows      = len(data) // n_cols

        t_key_ms = t_mkdir_ms = t_writer_ms = t_write_ms = 0
        n_writer_opens = 0
        data_offset = 0

        while data_offset < n_rows:
            # Derive current timestamp from original start + rows already written.
            # Computed from data_offset each iteration — no accumulated drift.
            now_s = int(start_s + data_offset * hop_s_float)

            t0k = _ticks_ms()
            current_key  = _epoch_to_partition_key(now_s, gran)
            part_start_s = _partition_start_epoch(current_key)
            part_end_s   = part_start_s + dur_s
            t_key_ms    += _ticks_diff(_ticks_ms(), t0k)

            last_key = self._last_partition.get(resource)
            if last_key is not None and last_key > current_key:
                raise ValueError(
                    "Out-of-order write: last partition {} > current {}".format(
                        last_key, current_key))
            self._last_partition[resource] = current_key

            # Rows fitting in this partition: row i is at now_s + i*hop_s_float,
            # which belongs here iff that timestamp < part_end_s.
            # int() truncation is correct: if gap is exactly N hops, row N
            # starts at part_end_s and belongs to the next partition.
            rows_remaining  = n_rows - data_offset
            rows_in_current = min(rows_remaining,
                                  int((part_end_s - now_s) / hop_s_float))

            # Guard: should not be <= 0 since now_s < part_end_s (timezone fix
            # ensures _parts_to_epoch is UTC-correct), but raise if it happens.
            if rows_in_current <= 0:
                raise ValueError(
                    "append_data: rows_in_current={} at offset={}/{} now_s={} part_end_s={}".format(
                        rows_in_current, data_offset, n_rows, now_s, part_end_s))

            first_row = int((now_s - part_start_s) / hop_s_float)
            pdir = _partition_dir(self._base, resource, gran,
                                  current_key[0], current_key[1], current_key[2],
                                  current_key[3], current_key[4])
            t0m = _ticks_ms()
            _makedirs(pdir)
            t_mkdir_ms += _ticks_diff(_ticks_ms(), t0m)

            # Use existing file if one already exists, else create with first_row.
            # Try the expected path first to avoid listdir on every call.
            expected_path = _data_file(pdir, resource, first_row)
            try:
                os.stat(expected_path)
                path = expected_path
            except OSError:
                existing = _find_data_files(pdir, resource)
                path = existing[0][1] if existing else expected_path

            t0o = _ticks_ms()
            w = delta_simple9.Writer(path, n_cols, DS9_CHUNK_SIZE)
            n_writer_opens += 1
            t_writer_ms += _ticks_diff(_ticks_ms(), t0o)

            t0w = _ticks_ms()
            w.write_rows(data, data_offset * n_cols, rows_in_current)
            w.close()
            t_write_ms += _ticks_diff(_ticks_ms(), t0w)

            data_offset += rows_in_current

        if self.debug:
            print("[append_data] writer_opens={} key={}ms mkdir={}ms writer_open={}ms write={}ms".format(
                n_writer_opens, t_key_ms, t_mkdir_ms, t_writer_ms, t_write_ms))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_info(self, resource):
        cfg  = self._res(resource)
        gran = cfg['granularity']
        dur  = _partition_duration_s(gran)

        first_key = None
        last_key  = None
        n_partitions = 0
        for key, pdir in _enumerate_partitions(self._base, resource, gran):
            if first_key is None:
                first_key = key
            last_key = key
            n_partitions += 1

        start_s = _partition_start_epoch(first_key) if first_key else None
        end_s   = _partition_start_epoch(last_key) + dur if last_key else None

        # Convert platform epoch back to Unix epoch for the public API.
        return {
            'resource':     resource,
            'granularity':  gran,
            'hop_us':       cfg['hop'],
            'columns':      cfg['columns'],
            'n_partitions': n_partitions,
            'start_s':      (start_s + _EPOCH_OFFSET) if start_s is not None else None,
            'end_s':        (end_s   + _EPOCH_OFFSET) if end_s   is not None else None,
        }
