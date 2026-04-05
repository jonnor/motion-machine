import array
import random

class ReservoirSampler:
    def __init__(self, k=1000, cols=7, typecode='h', seed=None):
        self.k = k
        self.cols = cols
        self.typecode = typecode

        # Flat storage: k * cols
        self.data = array.array(typecode, [0] * (k * cols))

        self.n_seen = 0      # total rows processed
        self.n_filled = 0    # rows currently in reservoir (<= k)

        if seed is not None:
            random.seed(seed)

    def push(self, chunk):
        """
        chunk: array.array with flat row-major data
        length must be multiple of cols
        """
        total_vals = len(chunk)
        cols = self.cols

        if total_vals % cols != 0:
            raise ValueError("Chunk size must be multiple of cols")

        k = self.k
        data = self.data

        for i in range(0, total_vals, cols):
            self.n_seen += 1

            if self.n_filled < k:
                # Fill phase
                dest_row = self.n_filled
                self.n_filled += 1
            else:
                # Reservoir sampling
                j = random.randrange(self.n_seen)  # [0, n_seen)
                if j >= k:
                    continue
                dest_row = j

            # Copy row (in-place overwrite)
            base = dest_row * cols
            for c in range(cols):
                data[base + c] = chunk[i + c]

    def get_flat(self):
        """Return underlying array (may be partially filled)."""
        return self.data

    def get_rows(self):
        """Yield rows as tuples (only filled rows)."""
        cols = self.cols
        for r in range(self.n_filled):
            base = r * cols
            yield tuple(self.data[base + c] for c in range(cols))

    def __len__(self):
        return self.n_filled
