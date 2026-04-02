
import array
import npyfile
import time
import os

from microhive import MicroHive

def _ms():
    return time.ticks_ms()

def _diff_ms(t):
    return time.ticks_diff(time.ticks_ms(), t)

def load_accel_to_db(db, source_npy, base_epoch,
                     n_cols=3, sensor_name="accel", chunk_rows=32):

    buf      = array.array('h', [0] * (chunk_rows * n_cols))
    row_idx  = 0
    n_appends = 0

    def flush(n_rows):
        nonlocal n_appends
        ts = base_epoch + (row_idx - n_rows)
        t0 = _ms()
        db.append_data(sensor_name, buf, timestamp_s=ts)
        ts_t = time.gmtime(ts)
        print("  append {}: {:02d}:{:02d}:{:02d}  rows {}-{}  ({} ms)".format(
            n_appends, ts_t[3], ts_t[4], ts_t[5],
            row_idx - n_rows, row_idx - 1, _diff_ms(t0)))
        n_appends += 1

    with npyfile.Reader(source_npy) as reader:
        assert reader.shape[1] == n_cols and reader.typecode == 'h'
        print("Loading {} samples from {}".format(reader.shape[0], source_npy))

        for chunk in reader.read_data_chunks(chunk_rows * n_cols):
            n_rows = len(chunk) // n_cols
            for i in range(len(chunk)):
                buf[i] = chunk[i]
            row_idx += n_rows
            flush(n_rows)

    print("Done. {} rows, {} appends".format(row_idx, n_appends))

def _rmdir_recursive(path):
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for e in entries:
        full = "{}/{}".format(path, e)
        try:
            os.listdir(full)
            _rmdir_recursive(full)
            os.rmdir(full)
        except OSError:
            try:
                os.remove(full)
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass

def main():
    path = 'data/pamap2_features.npy'

    columns = [
        'orient_x',
        'orient_y',
        'orient_z',
        'sma',
        'mean_x',
        'mean_y',
        'mean_z',
    ]

    database_dir = '/tsdb'
    _rmdir_recursive(database_dir)
    
    db = MicroHive(database_dir, {
        'metrics': {
            'hop': 1_000_000,
            'columns': columns,
            'dtype': 'int16',
            'granularity': 'hour',
        }
    })


    base_epoch = 1775052566 # 01.04.2026
    load_accel_to_db(db, path, sensor_name='metrics', n_cols=len(columns), base_epoch=base_epoch,
        chunk_rows=100)


if __name__ == '__main__':
    main()
