

"""
Accelerometer HAR feature extraction for MicroPython
Reads raw int16 .npy (N, 3), writes features .npy (n_windows, n_features)

Features per window (n_features = 7):
  0-2 : orientation x, y, z  (int16, scaled by orient_scale)
  3   : sma                   (int16, scaled by sma_scale, gravity-removed)
  4   : mean_x                (int16, raw ADC units)
  5   : mean_y
  6   : mean_z
"""

import array
import math
import npyfile

from application import Application


# ── Main processing ────────────────────────────────────────────────────────────
def load_file(app, input_path, chunk_rows=25, columns=3) -> int:
    """Returns number of windows written."""

    chunk_count = 0

    with npyfile.Reader(input_path) as reader:
        shape = reader.shape
        assert len(shape) == 2 and shape[1] == columns, \
            f"Expected (N, {columns}) array, got {shape}"
        assert reader.typecode == 'h', \
            f"Expected int16 ('h'), got '{reader.typecode}'"

        n_samples = shape[0]

        chunk_items = chunk_rows * columns
        for chunk in reader.read_data_chunks(chunk_items):

            # FIXME: specify starting time for the data
            app.process_accelerometer(chunk)
            chunk_count += 1

    return chunk_count



if __name__ == '__main__':

    app = Application()
    path = 'data/pamap2_25hz.npy'
    chunks = load_file(app, path)


