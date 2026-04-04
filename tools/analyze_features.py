"""
analyze_features.py - Analyze delta distributions and compression suitability
of int16 feature data stored in a .npy file.

Usage:
    python3 analyze_features.py <path_to_features.npy> [column_names...]

Example:
    python3 analyze_features.py data/pamap2_features.npy orient_x orient_y orient_z sma mean_x mean_y mean_z
"""

import sys
import numpy as np

def analyze(path, col_names=None):
    data = np.load(path)
    print("Shape: {}  dtype: {}".format(data.shape, data.dtype))

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_rows, n_cols = data.shape

    if col_names is None or len(col_names) != n_cols:
        col_names = ["col{}".format(c) for c in range(n_cols)]

    # Cast to int16 as the compressor would see it
    if data.dtype != np.int16:
        clipped = np.clip(data, -32768, 32767).astype(np.int16)
        clip_frac = np.mean(data != clipped)
        if clip_frac > 0:
            print("WARNING: {:.1f}% of values clipped to int16 range".format(clip_frac * 100))
        data = clipped

    print("\n{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}".format(
        "column", "min", "max", "mean_d", "med_d", "max_d", "bits_typ", "ratio_est"))
    print("-" * 82)

    raw_total  = n_rows * n_cols * 2
    comp_total = 0

    for c, name in enumerate(col_names):
        col    = data[:, c].astype(np.int32)
        deltas = np.abs(np.diff(col))

        mean_d  = deltas.mean()
        med_d   = np.median(deltas)
        max_d   = deltas.max()
        p95_d   = np.percentile(deltas, 95)

        # Zigzag encoding: delta d -> (d<<1)^(d>>16) & 0x1FFFF
        # Bits needed for p95 delta after zigzag
        zz_p95  = int(p95_d * 2)   # zigzag of positive delta is d*2
        bits    = max(1, int(zz_p95).bit_length())

        # Estimate Simple9 compression ratio for this bit width:
        # Simple9 packs floor(28/bits) values per 32-bit word
        vals_per_word = max(1, 28 // bits)
        # bits per stored value = 32 / vals_per_word
        bits_stored   = 32 / vals_per_word
        ratio_est     = 16 / bits_stored   # vs raw 16-bit

        # Estimate compressed size for this column
        comp_col = n_rows * bits_stored / 8
        comp_total += comp_col

        print("{:<14} {:>8} {:>8} {:>8.1f} {:>8.1f} {:>8} {:>10} {:>8.2f}x".format(
            name[:14],
            col.min(), col.max(),
            mean_d, med_d, max_d,
            "~{}b".format(bits),
            ratio_est))

    print("-" * 82)
    print("\nRaw size:       {:>8} bytes  ({:.1f} KB)".format(raw_total, raw_total/1024))
    print("Est comp size:  {:>8.0f} bytes  ({:.1f} KB)  (excl. headers)".format(
        comp_total, comp_total/1024))
    print("Est ratio:      {:>8.2f}x".format(raw_total / comp_total))

    # Recommendation
    print("\nRecommendation:")
    overall_ratio = raw_total / comp_total
    if overall_ratio > 1.5:
        print("  Good candidate for compression (est {:.1f}x).".format(overall_ratio))
    elif overall_ratio > 0.9:
        print("  Marginal benefit. Consider larger chunk_size to reduce header overhead.")
    else:
        print("  Compression makes data LARGER. Store raw int16 instead.")
        print("  Columns with large deltas (>~200) don't compress well with Simple9.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2:] if len(sys.argv) > 2 else None)
