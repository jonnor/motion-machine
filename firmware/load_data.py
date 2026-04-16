
"""
Load accelerometer and process it
"""

import array
import math
import npyfile

from application import Application


# ── Main processing ────────────────────────────────────────────────────────────
def load_file(app, input_path, columns=3, start_time=None) -> int:
    """Returns number of windows written."""

    chunk_count = 0

    if start_time is None:
        start_time = 1776296788 - (3600 * 24 * 10)

    chunk_rows = app.hop_length

    timestamp = start_time
    dt = 1.0 / app.samplerate
    with npyfile.Reader(input_path) as reader:
        shape = reader.shape
        assert len(shape) == 2 and shape[1] == columns, \
            f"Expected (N, {columns}) array, got {shape}"
        assert reader.typecode == 'h', \
            f"Expected int16 ('h'), got '{reader.typecode}'"

        n_samples = shape[0]

        chunk_items = chunk_rows * columns
        for chunk in reader.read_data_chunks(chunk_items):
            if len(chunk) < chunk_items:
                # end-of-file
                break

            # FIXME: specify starting time for the data
            app.process_accelerometer(chunk, timestamp=timestamp)
            chunk_count += 1
            timestamp += (dt * chunk_rows)

    return chunk_count



if __name__ == '__main__':

    app = Application()

    filter_path = 'firmware/orientation_lowpass.json'
    app.load_gravity_filter(filter_path)

    model_path = 'firmware/pamap2.trees.csv'
    app.load_model(model_path)

    data_path = 'data/pamap2_25hz.npy'
    chunks = load_file(app, data_path)


