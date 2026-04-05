
# Timeseries database

Must

- push raw accelerometer data, integrate window computations

Should

- Add end-to-end test for data processing, taking from pamap2_25hz
- Add link/button for "Open in Jupyter"

Nice

- push acceleration data over HTTP API

Demo maybe

- add some liveness indicator to watch, ex battery voltage on screen
- handle button press. Enable POWERON IRQ in AXP2101, IQR handler on pin 21
- 


# Improving compression

Limiting range slightly of int16 input data to, `[-16384, 16383]`
should make it possible to fit into 16 bit after delta-zigzag.
Maybe extending to 64-bit, Simple9b.
But with a selector, then can only fit 4x15 integers in 64 bits.



# Compression

delta-zigzag-simple9b did not compress PAMAP2 features much. Net loss.
delta-zigzag-simple9b. 406 kB in 18 seconds. Only 22 kB/s.

Would likely need to add quantization to get compression benefits.

raw16. 396 kB in 3 seconds. 132 kB/s.
Tested maximum for reads otherwise is around 210 kB/s.


```
mpremote run tests/bench_microhive.py
MicroHive benchmark  cols=7  hop=1000000us  gran=hour
platform: esp32

--- raw filesystem ---
  open+write+flush+close 1KB x50: 3402ms  (68.0 ms/op)
  open+read+close 1KB x50:        388ms  (7.8 ms/op)

--- append ---
  append 1h x7cols chunk=100: 2393ms total  1.5 rows/ms  66.5 ms/call
  append 1h x7cols chunk=1000: 1584ms total  2.3 rows/ms  396.0 ms/call

--- query (after 1h write, chunk=100) ---
  append 1h x7cols chunk=100: 2463ms total  1.5 rows/ms  68.4 ms/call
  query  3600rows chunk=64: 736ms  4.9 rows/ms  OK
  query  3600rows chunk=256: 738ms  4.9 rows/ms  OK
  query  3600rows chunk=512: 728ms  4.9 rows/ms  OK

--- query raw (row-major) codec ---
  query  3600rows chunk=256: 73ms  49.3 rows/ms  OK
--- query raw (col-major) codec ---
  query  3600rows chunk=256: 392ms  9.2 rows/ms  OK

```


# Running emlearn on JupyterLite

emlearn 0.23.2 provides a Python-only wheel.
Can be installed with JupyterLite.
Can use .convert(format='csv')



# Running on T-Watch S3

Standard image seems to work. Note: use the `.bin`.

esptool.py --baud 460800 write_flash -z 0x0 Downloads/ESP32_GENERIC_S3-20251209-v1.27.0.bin

How much space on filesystem?

```
import os
stats = os.statvfs('/')
block_size = stats[0]
total_blocks = stats[2]
free_blocks = stats[3]
print(f"Total: {block_size * total_blocks / 1024:.1f} KB")
print(f"Free:  {block_size * free_blocks / 1024:.1f} KB")
```

Got
```
14324.0 KB
```

BMA423 supports 25 Hz samplerate - not 20 Hz.

axp2101 looks to work

# Opening notebook with JupyterLite

there is a fromURL parameter for this
https://jupyterlite.readthedocs.io/en/stable/howto/content/open-url-parameter.html

But it does not seem to be supported on the scikit-learn instance
http://scikit-learn.org/stable/lite/lab/?fromURL=http://github.com/jakevdp/PythonDataScienceHandbook/raw/refs/heads/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb

But it does seem to be working on the Jupyter.org one
https://jupyter.org/try-jupyter/lab/index.html?fromURL=https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb

http://jupyter.org/try-jupyter/lab/index.html?fromURL=http://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb

Does it work with a local URL? Yes! As long as has CORS
http://jupyter.org/try-jupyter/lab/index.html?fromURL=http://localhost:8080/notebooks/Untitled.ipynb

And with on-device URL? YES!
http://jupyter.org/try-jupyter/lab/index.html?fromURL=http://192.168.87.141/files/Untitled.ipynb


# Loading data with JupyterLite

! works only with HTTP - not with HTTPS
Because our device uses HTTP, browser restriction

http://scikit-learn.org/stable/lite/lab/

```
import io
from datetime import datetime, timezone
from pyodide.http import pyfetch
import numpy as np

async def query(host, port, resource, start_dt, end_dt, chunk_rows=600):
    """
    Call GET /query and return a 2D numpy array via numpy.load.
    start_dt / end_dt: datetime objects (UTC).
    """
    start_s = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_s   = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    url = 'http://{}:{}/query?resource={}&start={}&end={}&chunk_rows={}'.format(
        host, port, resource, start_s, end_s, chunk_rows)

    response = await pyfetch(url)
    data = await response.bytes()

    return np.load(io.BytesIO(data))

start_dt = datetime.fromisoformat('2025-06-01T05:00:00')
end_dt   = datetime.fromisoformat('2025-06-01T07:00:00')

host = '10.126.225.242'
resource = 'sensor'
port = 80
arr = await query(host, port, resource, start_dt, end_dt)

arr
```

# emlearn-micropython WASM build and load

https://github.com/emlearn/emlearn-micropython/tree/gh-pages/builds/v0.10.1/ports/webassembly


```
curl -L -o frontend/micropython.wasm https://github.com/emlearn/emlearn-micropython/raw/refs/heads/gh-pages/builds/v0.10.1/ports/webassembly/micropython.wasm
curl -L -o frontend/micropython.mjs https://github.com/emlearn/emlearn-micropython/raw/refs/heads/gh-pages/builds/v0.10.1/ports/webassembly/micropython.mjs
```

!! this micropython.wasm is 1.1 MB. Whereas the default PyScript one is just 300 kB?


When running with emlearn-micropython build
```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ImportError: no module named 'pathlib'
```

pathlib is not included by default?
https://github.com/micropython/micropython-lib/tree/master/python-stdlib/pathlib

Where is the MicroPython configuration for the standard PyScript build?

There is a `pyscript` variant in upstream MicroPython. Has a bunch of modules included
https://github.com/micropython/micropython/blob/master/ports/webassembly/variants/pyscript/manifest.py


Trying to rebuild with VARIANT=pyscript
```
cp /home/jon/projects/emlearn-micropython/dist/ports/webassembly/micropython.* ./frontend/
```

NOTE: needs CFLAGS_EXTRA in webassembly port. `webassembly-extra-cflags` in `jonnor/micropython`

Now it works :)
