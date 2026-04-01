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

import struct
from microdot import Microdot, Response, send_file

from microhive import TYPECODE, _enumerate_partitions, _partition_start_epoch, _partition_duration_s
from files import cors, apply_cors
import files

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

    def __iter__(self):
        return self

    def _build_iter(self):
        total_rows = int((self._end_s - self._start_s) * 1_000_000) // self._hop_us

        # First chunk: .npy header
        yield _npy_header(total_rows, self._n_cols)

        # Subsequent chunks: raw bytes per query chunk
        for chunk in self._db.get_timerange(
                self._resource, self._start_s, self._end_s,
                chunk_rows=self._chunk_rows):
            yield bytes(chunk)

    def __next__(self):
        if self._iter is None:
            self._iter = self._build_iter()
        try:
            return next(self._iter)
        except StopIteration:
            raise

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def add_routes(app, db):
    """Register MicroHive routes on an existing Microdot app."""

    @app.route('/info', methods=["GET", "OPTIONS"])
    @cors()
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

    @app.route('/query', methods=["GET", "OPTIONS"])
    @cors()
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
        apply_cors(r)
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

    import asyncio
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(host='0.0.0.0', port=80, debug=True):
    import asyncio
    from microhive import MicroHive

    db = MicroHive('/mw_rw', {
        'sensor': {
            'hop': 1_000_000,
            'columns': ['sensor_a', 'sensor_b'],
            'dtype': 'int16',
            'granularity': 'hour',
        }
    })

    app = Microdot()
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

    # TODO: use static/ prefix - to avoid colliding with api endpoints
    @app.get('/static/<path:path>')
    async def static(request, path):
        return send_file('frontend/' + path, max_age=MAX_AGE)


    # Actually start server
    async def _startup():
        await _connect_wifi()
        print('MicroHive HTTP server on {}:{}'.format(host, port))
        print('  GET /query?resource=<n>&start=<epoch>&end=<epoch>[&chunk_rows=<n>]')
        await app.start_server(host=host, port=port, debug=debug)

    asyncio.run(_startup())


if __name__ == '__main__':
    main()
