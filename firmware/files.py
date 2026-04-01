
import os
from microdot import Microdot, Response, send_file

DEFAULT_CHUNK_SIZE = 1024


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

    @app.get('/files')
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

    @app.get('/files/<path:path>')
    def read_file(request, path):
        full_path = safe_path(base_dir, path)
        if not full_path:
            return {'error': 'Invalid path'}, 400
        try:
            os.stat(full_path)
        except OSError:
            return {'error': 'Not found'}, 404

        chunk_size = get_chunk_size(request, default_chunk_size)
        return Response(
            body=file_generator(full_path, chunk_size),
            headers={'Content-Type': 'application/octet-stream'}
        )

    @app.put('/files/<path:path>')
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

    @app.delete('/files/<path:path>')
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


# ── Usage examples ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = Microdot()

    MAX_AGE = 1 # set longer in production
    def on_changed(path, event):
        print(f'File {event}: {path}')

    add_routes(app, base_dir='data/', on_file_changed=on_changed)

    @app.get('/')
    async def index(request):
        return send_file('frontend/files_example.html')

    # TODO: use static/ prefix - to avoid colliding with api endpoints
    @app.get('/<path:path>')
    async def static(request, path):
        return send_file('frontend/' + path, max_age=MAX_AGE)

    app.run(host='0.0.0.0', port=5000)
