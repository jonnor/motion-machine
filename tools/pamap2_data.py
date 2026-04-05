
import os
import os.path

import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import numpy

def save_compressed_sequence(data, out_path):

    # Select only data from one IMU, 3 or 6 axis
    columns = [
        'hand_acceleration_6g_x',
        'hand_acceleration_6g_y',
        'hand_acceleration_6g_z',
        #'hand_gyro_x',
        #'hand_gyro_y',
        #'hand_gyro_z',
    ]
    sel = data[columns].reset_index()

    # Downsample to 25hz
    freq = '40ms'
    resampled = (
        sel.groupby('subject', observed=True)
        .apply(lambda g: g.set_index('time').resample(freq).mean())
        .reset_index()
    )

    # Pretend it is all from one subject
    # and concatenate in time
    resampled['time'] = pandas.timedelta_range(start=0, freq=freq, periods=len(resampled))
    resampled = resampled.drop(columns=['subject', 'time'])
    resampled


    # convert to int16
    # 32 is more than enough for the range of data
    scaled = (resampled / 16)*(32767/1)
    scaled = scaled.dropna().astype(numpy.int16)

    # Save as Parquet
    table = pa.Table.from_pandas(scaled, preserve_index=False)

    int_cols = [c for c, t in zip(table.schema.names, table.schema.types)
                if pa.types.is_integer(t)]

    pq.write_table(
        table,
        out_path,
        compression='gzip',
        compression_level=9,
        use_dictionary=False,
        write_statistics=False,
        column_encoding={col: 'DELTA_BINARY_PACKED' for col in int_cols},
        data_page_version='2.0',
        row_group_size=1*10_000,
        data_page_size=1*128*1024,
    )

def main():

    data_path = '/home/jon/projects/leaf-clustering-random-forests/data/processed/pamap2.parquet'
    out_path = 'data/pamap2_25hz.parquet'
    out_npy = 'data/pamap2_25hz.npy'

    data = pandas.read_parquet(data_path)

    durations = data.reset_index().groupby('subject', observed=True)['time'].max()
    print(durations)

    print(durations.sum())

    # convert
    save_compressed_sequence(data, out_path)

    compressed = os.path.getsize(out_path)
    print(compressed / 1e6, 'MB')

    # check that it can be loaded
    loaded = pandas.read_parquet(out_path)
    print(loaded.shape)
    print(loaded.columns)
    print(loaded.head())


    print('Save as npy')
    a = numpy.ascontiguousarray(loaded)
    numpy.save(out_npy, a, allow_pickle=False)

    compressed = os.path.getsize(out_npy)
    print(compressed / 1e6, 'MB')    

if __name__ == '__main__':
    main()





