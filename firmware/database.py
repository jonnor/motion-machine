"""
server.py - HTTP API for MicroHive using Microdot

Endpoint:
    GET /query?resource=<n>&start=<epoch_s>&end=<epoch_s>[&chunk_rows=<n>]

Returns the query result as a streaming .npy file (application/octet-stream).
The .npy header is written directly (no npyfile.Writer dependency) and data
chunks are streamed as raw bytes, one per get_timerange yield.

Usage:
    import server
    server.run(db, host='0.0.0.0', port=80)

MicroPython note:
    Async generator functions (async def + yield) are not supported in MicroPython.
    Streaming uses a class-based generator compatible with both platforms.
"""

import os
import struct
import asyncio
import time
import array
import gc

from microdot import Microdot, Response, send_file

from microhive import TYPECODE, _enumerate_partitions, _partition_start_epoch, _partition_duration_s
import files

gc.collect()

# ---------------------------------------------------------------------------
# Minimal .npy header writer (little-endian int16, C order, 2D)
# Avoids needing npyfile.Writer to accept a file-like object.
# ---------------------------------------------------------------------------

def _npy_header(n_rows, n_cols):
    """Return .npy v1.0 header bytes for a (n_rows, n_cols) int16 array."""
    descriptor = "{'descr': '<i2', 'fortran_order': False, 'shape': (%d, %d), }" % (n_rows, n_cols)
    # Total = 10 bytes fixed prefix + descriptor + '\n', padded to multiple of 64
    desc_bytes = descriptor.encode('latin-1')
    total = 10 + len(desc_bytes) + 1
    pad = (64 - (total % 64)) % 64
    desc_bytes = desc_bytes + b' ' * pad + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(desc_bytes)) + desc_bytes

# ---------------------------------------------------------------------------
# Sync class-based generator — MicroPython compatible streaming response
# ---------------------------------------------------------------------------

class _NpyStreamGenerator:
    """
    Streams a MicroHive query result as a .npy file, one bytes chunk per yield:
      - First yield: .npy header (shape + dtype descriptor)
      - Subsequent yields: raw int16 bytes for each chunk_rows block of data
    """

    def __init__(self, db, resource, start_s, end_s, chunk_rows, n_cols, hop_us):
        self._db         = db
        self._resource   = resource
        self._start_s    = start_s
        self._end_s      = end_s
        self._chunk_rows = chunk_rows
        self._n_cols     = n_cols
        self._hop_us     = hop_us
        self._iter       = None
        self._bytes_sent = 0

    def __iter__(self):
        return self

    def _build_iter(self):
        total_rows = int((self._end_s - self._start_s) * 1_000_000) // self._hop_us
        print('stream-gen-prep', total_rows)

        # First chunk: .npy header
        yield _npy_header(total_rows, self._n_cols)

        # Subsequent chunks: raw bytes per query chunk
        for chunk in self._db.get_timerange(
                self._resource, self._start_s, self._end_s,
                chunk_rows=self._chunk_rows):
            self._bytes_sent += len(chunk)
            print('stream-generator-chunk', len(chunk), self._bytes_sent)
            yield bytes(chunk)

    def __next__(self):
        if self._iter is None:
            self._iter = self._build_iter()
        try:
            return next(self._iter)
        except StopIteration:
            print('stream-generator-stop')
            raise

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def add_routes(app, db):
    """Register MicroHive routes on an existing Microdot app."""

    @app.get('/info')
    async def info(request):
        print('info hit')

        resource = request.args.get('resource')

        if not resource:
            return 'Missing resource', 400
        if resource not in db._resources:
            return 'Unknown resource: {}'.format(resource), 404

        cfg  = db._resources[resource]
        gran = cfg['granularity']
        dur  = _partition_duration_s(gran)

        first_key = None
        last_key  = None
        n_partitions = 0
        for key, _ in _enumerate_partitions(db._base, resource, gran):
            if first_key is None:
                first_key = key
            last_key = key
            n_partitions += 1

        if first_key is None:
            return {'resource': resource, 'has_data': False}, 200

        start_s = _partition_start_epoch(first_key)
        end_s   = _partition_start_epoch(last_key) + dur

        return {
            'resource':     resource,
            'has_data':     True,
            'start':        start_s,
            'end':          end_s,
            'n_partitions': n_partitions,
            'granularity':  gran,
            'columns':      cfg['columns'],
            'hop_us':       cfg['hop'],
        }, 200

    @app.get('/query')
    async def query(request):
        print('query hit')

        resource   = request.args.get('resource')
        start_s    = request.args.get('start')
        end_s      = request.args.get('end')
        chunk_rows = int(request.args.get('chunk_rows', '600'))

        if not resource or not start_s or not end_s:
            return 'Missing resource, start or end', 400

        try:
            start_s = int(start_s)
            end_s   = int(end_s)
        except ValueError:
            return 'start and end must be integer epoch seconds', 400

        if resource not in db._resources:
            return 'Unknown resource: {}'.format(resource), 404

        range_seconds = end_s - start_s
        print('query', range_seconds)

        cfg    = db._resources[resource]
        n_cols = len(cfg['columns'])
        hop_us = cfg['hop']

        gen = _NpyStreamGenerator(db, resource, start_s, end_s,
                                  chunk_rows, n_cols, hop_us)

        print('return generator')
        r = Response(gen, 200, {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': 'attachment; filename="{}.npy"'.format(resource),
        })
        return r



# ---------------------------------------------------------------------------
# WiFi
# ---------------------------------------------------------------------------

async def _connect_wifi():
    """Connect to WiFi using credentials from secrets.py.
    No-op on non-device platforms where the network module is unavailable."""
    try:
        import network
    except ImportError:
        return

    import secrets

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        print('WiFi already connected:', wlan.ifconfig()[0])
        return

    print('Connecting to WiFi SSID: {} ...'.format(secrets.SSID))
    wlan.connect(secrets.SSID, secrets.PASSWORD)

    while not wlan.isconnected():
        await asyncio.sleep(0.5)

    print('WiFi connected:', wlan.ifconfig()[0])



async def accelerometer_task():

    print('accelerometer-start')

    from machine import SoftI2C, I2C, Pin
    import bma423

    # how many samples to wait before reading.
    # BMA423 has 1024 bytes FIFO, enough for 150+ samples
    accel_samples = 40
    samplerate = 25

    # pre-allocate buffers
    # raw data (bytes). n_samples X 3 axes X 2 bytes
    accel_buffer = bytearray(accel_samples*3*2)

    # setup sensor
    #i2c = SoftI2C(scl=11,sda=10)
    i2c = I2C(scl=11,sda=10)
    sensor = bma423.BMA423(i2c, addr=0x19)
    sensor.fifo_enable()
    sensor.set_accelerometer_freq(samplerate)
    sensor.fifo_clear() # discard any samples lying around in FIFO
    await asyncio.sleep(0.1)

    print('accelerometer-init-done')
    counter = 0
    while True:
        
        # wait until we have enough samples
        fifo_level = sensor.fifo_level()
        if fifo_level >= len(accel_buffer):
            
            # read data
            read_start = time.ticks_ms()
            sensor.fifo_read(accel_buffer)
            read_dur = time.ticks_diff(time.ticks_ms(), read_start)
            
            print('accelerometer-read', read_start/1000, fifo_level, read_dur)

        
        # limit how often we check
        await asyncio.sleep(0.100)


def run_xor(model_path):

    import emlearn_trees

    model = emlearn_trees.new(15, 1000, 10)

    # Load a CSV file with the model
    with open(model_path, 'r') as f:
        emlearn_trees.load_model(model, f)

    # run it
    max_val = (2**15-1) # 1.0 as int16
    examples = [
        array.array('h', [0, 0]),
        array.array('h', [max_val, max_val]),
        array.array('h', [0, max_val]),
        array.array('h', [max_val, 0]),
    ]

    out = array.array('f', range(model.outputs()))
    for ex in examples:
        model.predict(ex, out)    
        result = out[1] > 0.5
        print(list(ex), '->', list(out), ':', result)


async def predict_task():

    print('predict-start')

    while True:

        model_path = 'notebooks/xor_model.csv'

        # check if model exists       
        exists = False 
        try:
            os.stat(model_path)
            exists = True
        except OSError:
            print('predict-model-not-found', model_path)

        if exists:
            print('predict-run', model_path)
            try:
                run_xor(model_path)
            # prevent from existing the loop/retrying
            except Exception as e:
                print('predict-error', e)
            print('\n')

        # limit how often we check
        await asyncio.sleep(10.000)


async def status_task():

    print('status-start')
    from machine import Pin
    from axp2101 import AXP2101

    pmu = AXP2101()
    pmu.twatch_s3_poweron()

    # Power on the display backlight.
    backlight = Pin(45, Pin.OUT)
    backlight.on()

    print('status-init-done')

    while True:

        batt = pmu.get_battery_voltage()
        print('status-run', batt)

        # limit how often we check
        await asyncio.sleep(10.000)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(host='0.0.0.0', port=80, debug=True):
    import asyncio
    from microhive import MicroHive


    accel = asyncio.create_task(accelerometer_task())

    predict = asyncio.create_task(predict_task())

    status = asyncio.create_task(status_task())

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

    db = MicroHive(database_dir, {
        'metrics': {
            'hop': 1_000_000,
            'columns': columns,
            'dtype': 'int16',
            'granularity': 'hour',
        },
    })

    from cors import CORS

    app = Microdot()

    cors = CORS(app, allowed_origins='*',
                allow_credentials=False)

    add_routes(app, db)
    
    # File handling
    def on_file_changed(path, event):
        print(f'File {event}: {path}')


    files.add_routes(app, base_dir='notebooks/', on_file_changed=on_file_changed)


    # User interface
    MAX_AGE = 1 # XXX: set longer in production, for more efficient caching

    @app.get('/')
    async def index(request):
        return send_file('frontend/database_example.html')

    @app.get('/filebrowser')
    async def index(request):
        return send_file('frontend/files_example.html')

    # TODO: use static/ prefix - to avoid colliding with api endpoints
    @app.get('/static/<path:path>')
    async def static(request, path):
        return send_file('frontend/' + path, max_age=MAX_AGE)


    print('free:', gc.mem_free())
    gc.collect()
    print('free:', gc.mem_free())

    # Actually start server
    async def _startup():
        await _connect_wifi()
        print('MicroHive HTTP server on {}:{}'.format(host, port))
        await app.start_server(host=host, port=port, debug=debug)

    asyncio.run(_startup())


if __name__ == '__main__':
    main()
