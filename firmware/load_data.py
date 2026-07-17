
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
    last_progress_log = -1

    if start_time is None:
        start_time = 1776296788 - (3600 * 24 * 10)

    chunk_rows = app.hop_length
    timestamp = start_time
    dt = 1.0 / app.samplerate
    print('load-file-start', input_path)
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

            samples_processed = (chunk_rows * chunk_count)
            progress_percent = int(100 * (samples_processed/n_samples))
            if progress_percent != last_progress_log:
                print(f'load-file-progress p={progress_percent}%')
                last_progress_log = progress_percent

    total_samples = chunk_count*chunk_rows # XXX: a bit off, last read might be short
    print(f'load-file-done samples={total_samples}')
    return total_samples



if __name__ == '__main__':

    app = Application(verbose=1)

    filter_path = 'firmware/orientation_lowpass.json'
    app.load_gravity_filter(filter_path)

    model_path = 'firmware/pamap2.trees.csv'
    app.load_model(model_path)

    data_path = 'data/pamap2_25hz.npy'
    chunks = load_file(app, data_path)


