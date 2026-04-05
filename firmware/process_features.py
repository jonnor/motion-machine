
"""
Accelerometer HAR feature extraction for MicroPython
Reads raw int16 .npy (N, 3), writes features .npy (n_windows, n_features)

Features per window (n_features = 7):
  0-2 : orientation x, y, z  (int16, scaled by orient_scale)
  3   : sma                   (int16, scaled by sma_scale, gravity-removed)
  4   : mean_x                (int16, raw ADC units)
  5   : mean_y
  6   : mean_z
"""

import array
import math


class AccelConfig:
    def __init__(
        self,
        n_axes=3,
        sample_rate=20,
        window_sec=4,
        hop_sec=1,
        orient_scale=16384,   # 1.0  → 16384 in int16
        sma_scale=32,         # tune to your ADC range; keeps SMA in int16 range
    ):
        self.n_axes       = n_axes
        self.sample_rate  = sample_rate
        self.window_sec   = window_sec
        self.hop_sec      = hop_sec
        self.orient_scale = orient_scale
        self.sma_scale    = sma_scale

        self.window_samples = sample_rate * window_sec
        self.hop_samples    = sample_rate * hop_sec
        self.n_features     = 7   # orient x/y/z, sma, mean x/y/z


# ── Feature computation ────────────────────────────────────────────────────────

def compute_features(win: array.array, out: array.array, cfg: AccelConfig) -> None:
    """
    win : array.array('h'), len = window_samples * n_axes, row-major
    out : array.array('h'), len = n_features — written in place
    """
    n   = len(win) // cfg.n_axes
    na  = cfg.n_axes

    # ── Gravity estimate (per-axis mean) ──────────────────────────────────────
    sx = sy = sz = 0
    j = 0
    for _ in range(n):
        sx += win[j]; sy += win[j+1]; sz += win[j+2]
        j += na
    mx = sx / n
    my = sy / n
    mz = sz / n

    # ── Orientation: normalize → unit vector → scale to int16 ────────────────
    mag = math.sqrt(mx*mx + my*my + mz*mz)
    if mag == 0.0:
        mag = 1.0
    sc = cfg.orient_scale
    out[0] = int(mx / mag * sc)
    out[1] = int(my / mag * sc)
    out[2] = int(mz / mag * sc)

    # ── SMA: gravity-removed dynamic acceleration ─────────────────────────────
    sma_sum = 0
    j = 0
    for _ in range(n):
        sma_sum += abs(win[j] - mx) + abs(win[j+1] - my) + abs(win[j+2] - mz)
        j += na
    out[3] = int(sma_sum / n / cfg.sma_scale)

    # ── Raw means (gravity / posture context) ─────────────────────────────────
    out[4] = int(mx)
    out[5] = int(my)
    out[6] = int(mz)


# ── Window count helper ────────────────────────────────────────────────────────

def count_windows(n_samples: int, cfg: AccelConfig) -> int:
    if n_samples < cfg.window_samples:
        return 0
    return (n_samples - cfg.window_samples) // cfg.hop_samples + 1


# ── Main processing ────────────────────────────────────────────────────────────

def process(cfg: AccelConfig, input_path, output_path) -> int:
    """Returns number of windows written."""
    import npyfile

    with npyfile.Reader(input_path) as reader:
        shape = reader.shape
        assert len(shape) == 2 and shape[1] == cfg.n_axes, \
            f"Expected (N, {cfg.n_axes}) array, got {shape}"
        assert reader.typecode == 'h', \
            f"Expected int16 ('h'), got '{reader.typecode}'"

        n_samples = shape[0]
        n_windows = count_windows(n_samples, cfg)
        print(f"Samples: {n_samples}, windows: {n_windows}")

        if n_windows == 0:
            print("Not enough samples for even one window.")
            return 0

        # ── Pre-allocate buffers ───────────────────────────────────────────────
        ws       = cfg.window_samples
        na       = cfg.n_axes
        win_buf  = array.array('h', [0] * (ws * na))
        feat_buf = array.array('h', [0] * cfg.n_features)
        ring_cap = (ws + cfg.hop_samples) * na
        ring     = array.array('h', [0] * ring_cap)
        ring_len = 0

        with npyfile.Writer(output_path,
                            shape=(n_windows, cfg.n_features),
                            typecode='h') as outfile:

            window_count = 0
            hop_items    = cfg.hop_samples * na
            chunks       = reader.read_data_chunks(hop_items)

            for chunk in chunks:
                # Append chunk to ring
                for v in chunk:
                    ring[ring_len] = v
                    ring_len += 1

                # Emit windows while enough data is available
                while ring_len >= ws * na:
                    for i in range(ws * na):
                        win_buf[i] = ring[i]

                    compute_features(win_buf, feat_buf, cfg)
                    outfile.write_values(feat_buf, typecode='h')
                    window_count += 1

                    # Slide ring left by one hop
                    remaining = ring_len - hop_items
                    for i in range(remaining):
                        ring[i] = ring[i + hop_items]
                    ring_len = remaining

            print(f"Windows written: {window_count}")
            return window_count



if __name__ == '__main__':
    # ── Entry point ────────────────────────────────────────────────────────────────

    cfg = AccelConfig(

    )
    n = process(cfg,
        input_path='data/pamap2_20hz.npy',
        output_path='data/pamap2_features.npy',
    )

