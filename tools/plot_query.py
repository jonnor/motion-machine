
#!/usr/bin/env python3
"""
plot_query.py - Query MicroHive HTTP API and plot as timeseries with Plotly

Usage:
    python3 plot_query.py --resource sensor --start 2025-06-01T05:00:00 --end 2025-06-01T07:00:00
    python3 plot_query.py --resource sensor --start 2025-06-01T05:00:00 --end 2025-06-01T07:00:00 --host 192.168.87.141
"""

import argparse
import io
from datetime import datetime, timezone
import urllib.request

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_HOST = '192.168.87.141'
DEFAULT_PORT = 80

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def query(host, port, resource, start_dt, end_dt, chunk_rows=600):
    """
    Call GET /query and return a 2D numpy array via numpy.load.
    start_dt / end_dt: datetime objects (UTC).
    """
    start_s = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_s   = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    url = 'http://{}:{}/query?resource={}&start={}&end={}&chunk_rows={}'.format(
        host, port, resource, start_s, end_s, chunk_rows)

    print('Querying: {}'.format(url))
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print('Received {} bytes'.format(len(data)))

    return np.load(io.BytesIO(data))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(arr, columns, start_dt, end_dt, resource):
    """Plot each column as a separate subplot sharing the x-axis."""
    n_rows, n_cols = arr.shape
    assert n_cols == len(columns), "Column count mismatch"

    # Build timestamp axis
    total_s   = (end_dt - start_dt).total_seconds()
    hop_s     = total_s / n_rows
    timestamps = [start_dt.replace(tzinfo=timezone.utc).timestamp() + i * hop_s
                  for i in range(n_rows)]
    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]

    fig = make_subplots(
        rows=n_cols, cols=1,
        shared_xaxes=True,
        subplot_titles=columns,
        vertical_spacing=0.06,
    )

    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(x=times, y=arr[:, i].tolist(), name=col,
                       mode='lines', line=dict(width=1)),
            row=i + 1, col=1,
        )

    fig.update_layout(
        title='MicroHive: {} — {} to {}'.format(
            resource,
            start_dt.strftime('%Y-%m-%d %H:%M'),
            end_dt.strftime('%H:%M UTC'),
        ),
        height=300 * n_cols,
        showlegend=False,
    )
    fig.update_xaxes(title_text='Time (UTC)', row=n_cols, col=1)
    fig.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Query MicroHive and plot timeseries')
    parser.add_argument('--host',       default=DEFAULT_HOST)
    parser.add_argument('--port',       default=DEFAULT_PORT, type=int)
    parser.add_argument('--resource',   default='sensor')
    parser.add_argument('--start',      required=True,
                        help='UTC datetime, e.g. 2025-06-01T05:00:00')
    parser.add_argument('--end',        required=True,
                        help='UTC datetime, e.g. 2025-06-01T07:00:00')
    parser.add_argument('--columns',    nargs='+',
                        help='Column names (default: sensor_a sensor_b)',
                        default=['sensor_a', 'sensor_b'])
    parser.add_argument('--chunk-rows', default=600, type=int)
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt   = datetime.fromisoformat(args.end)

    arr = query(args.host, args.port, args.resource, start_dt, end_dt, args.chunk_rows)
    print('Shape: {}  dtype: {}'.format(arr.shape, arr.dtype))
    plot(arr, args.columns, start_dt, end_dt, args.resource)


if __name__ == '__main__':
    main()
