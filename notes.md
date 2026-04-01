
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
