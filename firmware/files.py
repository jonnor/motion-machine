
import os
from microdot import Microdot, Response, send_file

DEFAULT_CHUNK_SIZE = 1024 # LittleFS block size is 4kB


def apply_cors(res, allow_origin='*', expose_headers=None):

    # ✅ Add CORS headers (do NOT overwrite existing ones)
    if "Access-Control-Allow-Origin" not in res.headers:
        res.headers["Access-Control-Allow-Origin"] = allow_origin

    if "Access-Control-Allow-Methods" not in res.headers:
        res.headers["Access-Control-Allow-Methods"] = \
            "GET, POST, PUT, DELETE, OPTIONS"

    # Echo requested headers if present
    req_headers = None # FIXME request.headers.get("Access-Control-Request-Headers")
    if req_headers:
        res.headers["Access-Control-Allow-Headers"] = req_headers
    elif "Access-Control-Allow-Headers" not in res.headers:
        res.headers["Access-Control-Allow-Headers"] = \
            "Content-Type, Authorization"

    # Optional: expose headers
    if expose_headers and "Access-Control-Expose-Headers" not in res.headers:
        res.headers["Access-Control-Expose-Headers"] = \
            ", ".join(expose_headers)


def cors(allow_origin="*", expose_headers=None):
    def decorator(handler, *dargs, **dkwargs):

        def wrapped(request, *args, **kwargs):

            # ✅ Handle preflight
            if request.method == "OPTIONS":
                res = Response(status_code=204)
            else:
                res = handler(request, *args, **kwargs)

                type_name = type(res).__name__
                if 'generator' in type_name:
                    # NOTE: for generators, apply_cors must be called manually
                    return res

                # Normalize to Response
                if not isinstance(res, Response) :
                    res = Response(*res)

            apply_cors(res, allow_origin, expose_headers)
            return res

        return wrapped

    return decorator


def safe_path(base_dir, path):
    """Resolve path within base_dir, blocking traversal attacks"""
    full = base_dir.rstrip('/') + '/' + path.lstrip('/')
    if not full.startswith(base_dir):
        return None
    return full


def file_generator(path, chunk_size):
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def stream_request_body(request, path, chunk_size):
    with open(path, 'wb') as f:
        while True:
            chunk = request.stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def get_chunk_size(request, default):
    try:
        return int(request.args.get('chunk_size', default))
    except (ValueError, TypeError):
        return default


def add_routes(app, base_dir='/files', default_chunk_size=DEFAULT_CHUNK_SIZE, on_file_changed=None):
    """
    Register file API routes on a Microdot app.

    Args:
        app:                Microdot app instance
        base_dir:           Root directory for all file operations (e.g. '/sd', '/flash')
        default_chunk_size: Default read/write chunk size in bytes; overridable per-request
                            via ?chunk_size=N query parameter
        on_file_changed:    Optional callback(path, event) called after write or delete.
                            event is 'write' or 'delete'.

    Endpoints:
        GET    /files              List files in base_dir
        GET    /files/<path>       Stream a file (supports ?chunk_size=N)
        PUT    /files/<path>       Write/overwrite a file (supports ?chunk_size=N)
        DELETE /files/<path>       Delete a file
    """

    def notify(path, event):
        if on_file_changed:
            try:
                on_file_changed(path, event)
            except Exception as e:
                print('on_file_changed error:', e)

    @app.route('/files', methods=["GET", "OPTIONS"])
    @cors()
    def list_files(request):
        try:
            files = []
            for entry in os.ilistdir(base_dir):
                name, ftype, *_ = entry
                files.append({
                    'name': name,
                    'type': 'dir' if ftype == 0x4000 else 'file'
                })
            return {'files': files, 'base_dir': base_dir}
        except OSError as e:
            return {'error': str(e)}, 500

    @app.route('/files/<path:path>', methods=["GET", "OPTIONS"])
    @cors()
    def read_file(request, path=None):
        full_path = safe_path(base_dir, path)
        if not full_path:
            return {'error': 'Invalid path'}, 400
        try:
            os.stat(full_path)
        except OSError:
            print('file not found', path, full_path)
            return {'error': 'Not found'}, 404

        chunk_size = get_chunk_size(request, default_chunk_size)
        r = Response(
            body=file_generator(full_path, chunk_size),
            headers={'Content-Type': 'application/octet-stream'}
        )
        apply_cors(r) # manual for generators
        return r

    @app.route('/files/<path:path>', methods=["PUT", "OPTIONS"])
    @cors()
    def write_file(request, path):
        full_path = safe_path(base_dir, path)
        if not full_path:
            return {'error': 'Invalid path'}, 400

        parent = full_path.rsplit('/', 1)[0]
        if parent and parent != base_dir.rstrip('/'):
            try:
                os.mkdir(parent)
            except OSError:
                pass  # Already exists

        chunk_size = get_chunk_size(request, default_chunk_size)
        try:
            stream_request_body(request, full_path, chunk_size)
        except Exception as e:
            return {'error': str(e)}, 500

        notify(full_path, 'write')
        return {'status': 'ok', 'path': full_path}, 201

    @app.route('/files/<path:path>', methods=["DELETE", "OPTIONS"])
    @cors()
    def delete_file(request, path):
        full_path = safe_path(base_dir, path)
        if not full_path:
            return {'error': 'Invalid path'}, 400
        try:
            os.remove(full_path)
        except OSError:
            return {'error': 'Not found'}, 404

        notify(full_path, 'delete')
        return {'status': 'ok'}, 200


