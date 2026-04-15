
"""
Load accelerometer and process it
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

    filter_path = 'firmware/orientation_lowpass.json'
    app.load_gravity_filter(filter_path)

    data_path = 'data/pamap2_25hz.npy'
    chunks = load_file(app, data_path)


