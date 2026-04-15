"""
"""

import os
import struct
import asyncio
import time
import array
import gc
import json
import math

from microdot import Microdot, Response, send_file

from microhive import MicroHive
import microhive_api
import files

from sliding_window import SlidingWindow
from orientation import GravityEstimatorLowpass, normalize_gravity, compute_tilt

gc.collect()

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



async def accelerometer_task(app):

    print('accelerometer-start')

    from machine import SoftI2C, I2C, Pin
    import bma423

    # how many samples to wait before reading.
    # BMA423 has 1024 bytes FIFO, enough for 150+ samples
    accel_samples = 25
    samplerate = 25

    # pre-allocate buffers
    # raw data (bytes). n_samples X 3 axes X 2 bytes
    accel_array = array.array('h', (0 for _ in range(accel_samples*3)))
    accel_buffer = memoryview(accel_array)

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

            # process it
            app.process_accelerometer(accel_buffer)

        
        # limit how often we check
        await asyncio.sleep(0.100)


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

def setup_resources(samplerate=25):

    feature_columns = [
        'orient_mean_x',
        'orient_mean_y',
        'orient_mean_z',
        'motion_sma',
        'orient_std_x',
        'orient_std_y',
        'orient_std_z',
    ]

    class_columns = [
        'a',
        'b',
        'c',
        'd',
    ],

    resources = {}

    resources['features'] = {
        'hop': 1_000_000,
        'columns': feature_columns,
        'dtype': 'int16',
        'granularity': 'hour',
        'codec': 'raw',
    }

    resources['predictions'] = {
        'hop': 1_000_000,
        'columns': class_columns,
        'dtype': 'int16',
        'granularity': 'hour',
        'codec': 'raw',
    }

    resources['raw'] = {
        'hop': int(1_000_000/samplerate),
        'columns': ['acc_x', 'acc_y', 'acc_z'],
        'dtype': 'int16',
        'granularity': 'minute',
        'codec': 'raw',
    }

    return resources


class RunningStats:
    """Welford's online algorithm for mean, variance, and std dev."""

    def __init__(self, dims):
        self.dims = dims
        self.count = 0
        self.mean   = array.array('f', [0.0] * dims)
        self._M2    = array.array('f', [0.0] * dims)

    def update(self, x):
        self.count += 1
        for i in range(self.dims):
            delta = x[i] - self.mean[i]
            self.mean[i] += delta / self.count
            self._M2[i] += delta * (x[i] - self.mean[i])

    def variance(self, out=None):
        if out is None:
            out = array.array('f', [0.0] * self.dims)
        if self.count < 2:
            for i in range(self.dims):
                out[i] = 0.0
        else:
            for i in range(self.dims):
                out[i] = self._M2[i] / (self.count - 1)
        return out

    def std(self, out=None):
        out = self.variance(out)
        for i in range(self.dims):
            out[i] = math.sqrt(out[i])
        return out


class Application():
    def __init__(self, database_dir='tsdb', verbose=3):

        resources = setup_resources()
        self.db = MicroHive(database_dir, resources)

        samplerate = 25
        window_length =  int(samplerate * 4.0)
        hop_length = int(samplerate * 1.0)
        self.window = SlidingWindow(window_length, hop_length, 3)

        self.verbose = verbose

        # feature extraction
        self.n_features = 7
        self.features = array.array('h', (0 for _ in range(self.n_features)))

        # lazy-loaded
        self.predictions = None
        self.model = None
        self.gravity = None

    def load_model(self, path):

        import emlearn_trees
        model = emlearn_trees.new(15, 1000, 10)

        # Load a CSV file with the model
        with open(model_path, 'r') as f:
            emlearn_trees.load_model(model, f)

        self.predictions = array.array('f', range(model.outputs()))
        self.model = model

    def load_gravity_filter(self, path):

        with open(path) as f:
            config = json.loads(f.read())

        # FIXME: check samplerate
        #config['samplerate']

        coefficients = array.array('f', config['coefficients'])
        self.gravity = GravityEstimatorLowpass(coefficients)

    def process_window(self, win : array.array):

        # TODO: run orientation estimation, extract
        assert self.gravity is not None, 'gravity filter not loaded'

        orientation_stats = RunningStats(dims=3)
        orientation = array.array('f', (0 for i in range(3)))

        sma_sum = 0.0

        n_samples = len(win) // 3
        for i in range(n_samples):
            #sample += 1
            xyz = win[(i*3):(i*3)+3]
            gravity = self.gravity.update(xyz)
            orientation = normalize_gravity(gravity, out=orientation)
            #pitch, roll = compute_tilt(orientation)

            # summarize orientation
            orientation_stats.update(orientation)

            ax = win[(i*3)+0]
            ay = win[(i*3)+1]
            az = win[(i*3)+1]

            # compute the motion (linear-acceleration)
            # by subtracting gravity vector
            mx = ax - gravity[0]
            my = ay - gravity[1]
            mz = az - gravity[2]

            # compute Signal Magnitude Area (SMA)
            sma_sum += abs(mx) + abs(my) + abs(mz)

            # 

        # Assign outputs to feature vector - with scaling
        # TODO: support dynamic feature selection/order
        out = self.features

        # XXX: make sure to match order of column definition
        orient_scale = 1000
        # overall orientation
        orient_mean = orientation_stats.mean
        out[0] = int(orient_mean[0] * orient_scale)
        out[1] = int(orient_mean[1] * orient_scale)
        out[2] = int(orient_mean[2] * orient_scale)

        # overall energy
        sma_scale = 64
        out[3] = int((sma_sum / n_samples) * sma_scale)

        # variation in orientation

        # orientation_stats.std()
        orient_std = orientation_stats.std()
        out[4] = int(orient_std[0] * orient_scale)
        out[5] = int(orient_std[1] * orient_scale)
        out[6] = int(orient_std[2] * orient_scale)



    def process_accelerometer(self, accel):

        if self.verbose >= 2:
            print('process-accelerometer', len(accel))

        # store raw data
        self.db.append_data('raw', accel)

        # compute overlapped window
        window = self.window.push(accel)
        if window is not None:

            # extract features
            self.process_window(window)

            # store features
            self.db.append_data('features', self.features)

            if self.verbose >= 3:
                print('features', self.features)
            if self.model is not None:
                self.model.predict(self.features, self.predictions)

            # TODO: store predictions, at least for recent?


def add_routes(app, db, on_file_changed=None):

    files.add_routes(app, base_dir='notebooks/', on_file_changed=on_file_changed)

    microhive_api.add_routes(app, db)

    # User interface
    MAX_AGE = 1 # XXX: set longer in production, for more efficient caching

    @app.get('/')
    async def index(request):
        return send_file('frontend/database_example.html')

    @app.get('/filebrowser')
    async def index(request):
        return send_file('frontend/files_example.html')

    @app.get('/static/<path:path>')
    async def static(request, path):
        return send_file('frontend/' + path, max_age=MAX_AGE)



def main(host='0.0.0.0', port=80, debug=True):

    state = Application()

    # Load processing configuration
    filter_path = 'orientation_lowpass.json'
    state.load_gravity_filter(filter_path)

    # Start the different tasks
    status = asyncio.create_task(status_task())

    accel = asyncio.create_task(accelerometer_task(state))

    # File handling
    def on_file_changed(path, event):
        print(f'File {event}: {path}')


    # Web server
    from cors import CORS
    app = Microdot()
    cors = CORS(app, allowed_origins='*',
                allow_credentials=False)

    add_routes(app, state.db, on_file_changed=on_file_changed)
    

    # Reduce memory pressure
    gc.collect()
    print('app-start', 'mem_free={}'.format(gc.mem_free()))

    # Actually start server
    async def _startup():
        await _connect_wifi()
        print('HTTP server on {}:{}'.format(host, port))
        await app.start_server(host=host, port=port, debug=debug)

    asyncio.run(_startup())


if __name__ == '__main__':
    main()
