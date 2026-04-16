
import os
import sys
import array
from application import Application


def parse_args(argv, config):
    """
    Parse --key=value or --key value style long-form arguments into config dict.
    Supported types are inferred from the default value in config.
    """
    i = 1
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith('--'):
            print("Unknown argument: " + arg)
            sys.exit(1)

        arg = arg[2:]  # strip --

        if '=' in arg:
            key, value = arg.split('=', 1)
        else:
            key = arg
            i += 1
            if i >= len(argv):
                print("Missing value for --" + key)
                sys.exit(1)
            value = argv[i]

        if key not in config:
            print("Unknown option: --" + key)
            sys.exit(1)

        default = config[key]
        if isinstance(default, int):
            try:
                config[key] = int(value)
            except ValueError:
                print("--" + key + " expects an integer, got: " + value)
                sys.exit(1)
        else:
            config[key] = value

        i += 1

    return config

def csv_batches(path, n):
    with open(path) as f:
        headers = next(f).rstrip('\n').split(',')
        batch = []
        for line in f:
            row = dict(zip(headers, line.rstrip('\n').split(',')))
            batch.append(row)
            if len(batch) == n:
                yield headers, batch
                batch = []
        if batch:
            yield headers, batch

def main():
    # Defaults
    config = {
        'input':         '',
        'output':        '',
        'samplerate':    25,
        'hop_length':    10,
        'window_length': 100,
    }

    here = os.path.abspath(os.path.dirname(__file__))

    parse_args(sys.argv, config)
    print(config)

    hop_length = int(config['hop_length'])
    window_length = int(config['window_length'])
    samplerate = int(config['samplerate'])


    app = Application(window_length=window_length, samplerate=samplerate, hop_length=hop_length)
    filter_path = os.path.join(here, 'orientation_lowpass.json')
    app.load_gravity_filter(filter_path)

    typecode = 'h'
    samples = array.array(typecode, (0 for _ in range(3*window_length)))
    sample_scale = 1.0 # 2**15-1

    # write feature column names
    feature_resource = app.db._resources['features']
    feature_columns = feature_resource['columns']
    headers = ['time'] + feature_columns
    with open(config['output'], 'w') as out_file:
        out_file.write(','.join(headers) + '\n')
    

        expect_columns = set(['time', 'acc_x', 'acc_y', 'acc_z'])
        batches = 0
        rows = 0
        time = 0.0
        dt = 1.0/samplerate
        for headers, batch in csv_batches(config['input'], hop_length):
            miss_columns = set(headers) - expect_columns
            assert miss_columns == set()

            assert len(batch) <= hop_length

            # convert to array
            for i, row in enumerate(batch):  
                row['time'] # not just, just check that it is there
                samples[(i*3)+0] = int(float(row['acc_x']) * sample_scale)
                samples[(i*3)+1] = int(float(row['acc_y']) * sample_scale)
                samples[(i*3)+2] = int(float(row['acc_z']) * sample_scale)

            # Compute features
            window = app.window.push(samples)
            if window is not None:
                app.process_window(window)

                # write features to output file
                out_row = [ time ] + list(app.features)
                out_file.write(','.join(str(v) for v in out_row) + '\n')
                rows += 1

            time += (len(batch) * dt)
            batches += 1

    print('time', time, 'batches', batches, 'rows', rows)
    #raise Exception("foo")

if __name__ == '__main__':
    main()

