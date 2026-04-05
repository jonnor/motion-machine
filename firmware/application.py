"""
"""

import os
import struct
import asyncio
import time
import array
import gc

from microdot import Microdot, Response, send_file

from microhive import MicroHive
import microhive_api
import files

from sliding_window import SlidingWindow

from process_features import compute_features, AccelConfig

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
        'orient_x',
        'orient_y',
        'orient_z',
        'sma',
        'mean_x',
        'mean_y',
        'mean_z',
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



class Application():
    def __init__(self, database_dir='tsdb'):

        resources = setup_resources()
        self.db = MicroHive(database_dir, resources)

        samplerate = 25
        window_length =  int(samplerate * 4.0)
        hop_length = int(samplerate * 1.0)
        self.window = SlidingWindow(window_length, hop_length, 3)

        # feature extraction
        self.n_features = 7
        self.features = array.array('h', (0 for _ in range(self.n_features)))

        # lazy-loaded
        self.predictions = None
        self.model = None

    def load_model(self, path):

        import emlearn_trees
        model = emlearn_trees.new(15, 1000, 10)

        # Load a CSV file with the model
        with open(model_path, 'r') as f:
            emlearn_trees.load_model(model, f)

        self.predictions = array.array('f', range(model.outputs()))
        self.model = model


    def process_accelerometer(self, accel):

        print('process-accelerometer', len(accel), accel)

        # store raw data
        self.db.append_data('raw', accel)

        # compute overlapped window
        cfg  = AccelConfig(sample_rate=25, hop_sec=1.0, window_sec=4.0)

        window = self.window.push(accel)
        if window is not None:

            # TODO: get rid of the AccelConfig, duplicated settings
            compute_features(window, self.features, cfg)

            # store features
            self.db.append_data('features', self.features)

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
