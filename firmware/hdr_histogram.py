"""
HdrHistogram — bounded-error percentile tracking in constant memory.

Tracks approximated percentiles of a value stream without storing individual samples.

Useful for latency tracking, sensor reading distributions,
or any long-tailed metric where the tail (p90, p99, etc) is of interest.

Key features:
  - Tracks non-negative integer-valued metrics (example: time in microseconds).
  - Pure-Python, works on MicroPython and CPython.
  - No external dependencies. Only needs standard library (math and array).
  - Fixed memory, set at construction time.
  - Constant relative error across the specified range - same precision
    at the tail as near the median.
  - O(1) record(): a few bit shifts and one array write, no allocation
  - Mergeable: multiple histograms can be combined.

Notes:
  - record() requires a non-negative integer within [0, max_value] and
    raises HdrRecordError otherwise -- scale float inputs (e.g. seconds)
    into an integer unit (e.g. microseconds) before recording
  - percentile()/percentiles()/mean() raise HdrEmptyHistogramError if
    called before any value has been recorded
"""

import math
from array import array


class HdrRecordError(ValueError):
    """Raised by record() for non-integer or out-of-range values."""
    pass


class HdrEmptyHistogramError(Exception):
    """Raised by percentile()/percentiles()/mean() when no values recorded."""
    pass


class HdrHistogram:
    """
    Pure-Python HDR Histogram. Stdlib only (math, array). Works on CPython
    and MicroPython. Fixed memory footprint, O(1) record, mergeable,
    constant relative error across the tracked range.
    """

    def __init__(self, max_value, sig_digits=3, bits=32):
        if bits == 16:
            self.typecode = 'H'
            self.max_count = 0xFFFF
            self.itemsize = 2
        elif bits == 32:
            self.typecode = 'I'
            self.max_count = 0xFFFFFFFF
            self.itemsize = 4
        else:
            raise ValueError("bits must be 16 or 32")

        if not isinstance(max_value, int) or max_value < 1:
            raise ValueError("max_value must be an integer >= 1")

        self.max_value = max_value
        self.sig_digits = sig_digits

        self.sub_bucket_count = 1
        while self.sub_bucket_count < 10 ** sig_digits:
            self.sub_bucket_count *= 2
        self.sub_bucket_half_count = self.sub_bucket_count // 2

        # log2(sub_bucket_half_count), computed manually (no int.bit_length()
        # on MicroPython)
        n = self.sub_bucket_half_count
        m = 0
        while n > 1:
            n >>= 1
            m += 1
        self.sub_bucket_half_count_magnitude = m

        self.bucket_count = 1
        smallest_untrackable = self.sub_bucket_count
        while smallest_untrackable <= self.max_value:
            smallest_untrackable <<= 1
            self.bucket_count += 1

        self.counts_len = (
            (self.bucket_count + 1) * self.sub_bucket_half_count
            + self.sub_bucket_half_count
        )
        self.counts = array(self.typecode, [0] * self.counts_len)
        self.total_count = 0

    # ---- index math ----
    def _get_bucket_index(self, value):
        smallest_untrackable = self.sub_bucket_count
        buckets_needed = 1
        while smallest_untrackable <= value:
            smallest_untrackable <<= 1
            buckets_needed += 1
        return buckets_needed - 1

    def _get_sub_bucket_index(self, value, bucket_index):
        return value >> bucket_index

    def _counts_index(self, bucket_index, sub_bucket_index):
        bucket_base_index = (bucket_index + 1) << self.sub_bucket_half_count_magnitude
        offset = sub_bucket_index - self.sub_bucket_half_count
        return bucket_base_index + offset

    def _value_from_index(self, bucket_index, sub_bucket_index):
        return sub_bucket_index << bucket_index

    # ---- public API ----
    def record(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise HdrRecordError("value must be an integer, got %r" % (value,))
        if value < 0 or value > self.max_value:
            raise HdrRecordError(
                "value %d out of range [0, %d]" % (value, self.max_value)
            )

        bucket_index = self._get_bucket_index(value)
        sub_bucket_index = self._get_sub_bucket_index(value, bucket_index)
        idx = self._counts_index(bucket_index, sub_bucket_index)

        if idx < 0:
            idx = 0
        if idx >= self.counts_len:
            idx = self.counts_len - 1

        if self.counts[idx] < self.max_count:
            self.counts[idx] += 1
        self.total_count += 1

    def _iter_values(self):
        # indices 0..sub_bucket_count-1 belong to bucket 0, which gets full
        # resolution (no half-range restriction)
        for sub_bucket_index in range(self.sub_bucket_count):
            c = self.counts[sub_bucket_index]
            if c:
                yield self._value_from_index(0, sub_bucket_index), c

        # every subsequent bucket only uses the upper half of the
        # sub-bucket range (the lower half would already be represented by
        # a smaller bucket)
        for bucket_index in range(1, self.bucket_count + 1):
            for sub_bucket_index in range(self.sub_bucket_half_count, self.sub_bucket_count):
                idx = self._counts_index(bucket_index, sub_bucket_index)
                if idx >= self.counts_len:
                    continue
                c = self.counts[idx]
                if c:
                    yield self._value_from_index(bucket_index, sub_bucket_index), c

    def percentile(self, p):
        return self.percentiles([p])[p]

    def percentiles(self, ps):
        """
        Compute multiple percentiles in a single pass over the counts
        array. Returns a dict {p: value}. Much cheaper than calling
        percentile() once per p, since the O(counts_len) walk is shared.
        """
        if self.total_count == 0:
            raise HdrEmptyHistogramError("no values recorded")

        result = {}
        # sort targets ascending so we can satisfy them in one forward pass
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        targets = [math.ceil((ps[i] / 100.0) * self.total_count) for i in order]

        cumulative = 0
        last_value = 0
        ti = 0
        for value, count in self._iter_values():
            cumulative += count
            last_value = value
            while ti < len(targets) and cumulative >= targets[ti]:
                result[ps[order[ti]]] = value
                ti += 1
            if ti >= len(targets):
                break

        # any remaining targets (e.g. p=100 beyond last populated slot)
        while ti < len(targets):
            result[ps[order[ti]]] = last_value
            ti += 1

        return result

    def mean(self):
        if self.total_count == 0:
            raise HdrEmptyHistogramError("no values recorded")
        total = 0
        for value, count in self._iter_values():
            total += value * count
        return total / self.total_count

    def merge(self, other):
        for i in range(self.counts_len):
            s = self.counts[i] + other.counts[i]
            self.counts[i] = s if s <= self.max_count else self.max_count
        self.total_count += other.total_count

    def reset(self):
        for i in range(self.counts_len):
            self.counts[i] = 0
        self.total_count = 0

    def count(self):
        return self.total_count

    def memory_bytes(self):
        return self.itemsize * self.counts_len

