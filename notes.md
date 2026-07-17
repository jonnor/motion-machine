
# Presentation-ready

The pieces we want from project

- Showing EDA in browser. Interactive analysis
- On-device inference with Random Forest
- Analyze data i Jupyter Lite via API. Potentially push back to device

#### Demo 1: On-device time-series database

Loading many hours of data from device. Various detail levels.

- UI. Add ability to select time-section, show raw data

#### Demo 2: On-device inference

- Compute class durations from classifier, store in database
- Visualize class durations in webUI

#### Demo 3: Clustering for EDA

- Import data onto device/database. Incl features
- Add PCA 2d projection of features to browser
- Allow selecting in PCA/scatterplot -> show on timeline
- In scatter and timeline

Bonus:

- Allow selecting features generally - not just PCA plots
- Use ReservoirSampler on-device.
To have a ready-to-use random sample of features.
Kind-of a cache. Say 1k samples a 10 features, approx 20 kB
- Clustering. Show PCA variance explained over components

#### Demo 4: Use data in other environments

- Add link/button for "Open in Jupyter"
- Record video

People can imagine what to do from there.
And explain can push data back to the device


#### General

Should

- Fix time partitioning when loading accelerometer data historically

Want

- UI. Scatterplot matrix over features. Time windows selection
- Few-shot learning using emlearn_neighbors
- Frontend. Cluster windows data using k-means. See on timeline.
Fix emlearn_kmeans for float and/or int16.

### Demo X: In-jupyter classifier training

?? Does har_train work in browser
Skipped!
Just show exiting to Jupyter notebook.
People can imagine what to do from there.
And explain can push data back to the device


### Demo X: In-browser classifier training

- Train model in-browser using MicroPython with emlearn_extratrees


## Cleanups

- Move PCA and scaler into emlearn-micropython.
Make clustering example, include ReservoirSampler
Demonstrate on a well-known dataset
- Move feature calculations into emlearn-micropython or emlearn-motion. HAR example?
- emlearn-micropyton. Run tests also in browser

## Later

Labeling

- UI. Label sections on timeline.

Maybe

- Features. Cross-axis information
- Features. Band energies. 4-5 bands
- Push acceleration data over HTTP API, for testing

Database

- Column selection/filtering.

# Workflow

- Data collection
- Explore. Cluster
- Label.
- Classifier training
- Reprocess entire. Visualize overall. Class proportions, timeline.

# Improving compression

Limiting range slightly of int16 input data to, `[-16384, 16383]`
should make it possible to fit into 16 bit after delta-zigzag.
Maybe extending to 64-bit, Simple9b.
But with a selector, then can only fit 4x15 integers in 64 bits.


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
