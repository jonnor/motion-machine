
# Timeseries database

- try fetching data from device using JupyterLite
- add frontend for fetching timeseries using API
- load some real data - compute metrics over overlapped windows and store. PAMAP2
- add rollup hooks, push raw accelerometer data, add window computations there

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
