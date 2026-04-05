import array


class SlidingWindow:
    def __init__(self, window_size: int, hop_size: int, n_cols: int, typecode: str = 'h') -> None:
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_cols = n_cols
        self._stride = window_size * n_cols
        self._hop_elems = hop_size * n_cols
        self._buf = array.array(typecode, [0] * self._stride)
        self._mv = memoryview(self._buf)  # elements match array typecode
        self._push_count = 0

    def push(self, chunk: array.array):
        """Shift window left by hop, append chunk at tail.
        Returns a memoryview into the internal buffer, or None until full.
        The view is only valid until the next push()."""
        for i in range(self._stride - self._hop_elems):
            self._buf[i] = self._buf[i + self._hop_elems]
        tail = self._stride - self._hop_elems
        for i in range(self._hop_elems):
            self._buf[tail + i] = chunk[i]
        self._push_count += 1
        return self._mv if self._push_count >= self.window_size // self.hop_size else None

    def ready(self) -> bool:
        """True once the window has been filled for the first time."""
        return self._push_count >= self.window_size // self.hop_size

    def reset(self) -> None:
        """Clear the buffer and reset fill state."""
        for i in range(self._stride):
            self._buf[i] = 0
        self._push_count = 0
