"""
scaler.py - StandardScaler for MicroPython
Modelled on sklearn.preprocessing.StandardScaler.

Input:  array.array typecode 'h' (int16)
Output: array.array typecode 'f' (float32)
Internal computation: float

Conventions match pca.py:
  - Flat array layout: row-major, shape (n_samples, n_features)
  - All output buffers allocated early and passed into functions
  - No hidden allocations during transform (if out= supplied)
"""

import math
import array


# ---------------------------------------------------------------------------
# Low-level helpers (shared conventions with pca.py)
# ---------------------------------------------------------------------------

def _alloc_f(n):
    """Allocate zero-filled float array of length n."""
    return array.array('f', (0.0 for _ in range(n)))

def _alloc_f2(rows, cols):
    """Allocate flat float array for a rows x cols matrix."""
    return array.array('f', (0.0 for _ in range(rows * cols)))


# ---------------------------------------------------------------------------
# Internal statistics computation
# ---------------------------------------------------------------------------

def _compute_mean(X, n_samples, n_feat, mean_out):
    """
    Column means of flat array X (int16 or float) into mean_out (float).
    mean_out must be pre-allocated, length n_feat.
    """
    for j in range(n_feat):
        mean_out[j] = 0.0
    for s in range(n_samples):
        base = s * n_feat
        for j in range(n_feat):
            mean_out[j] += float(X[base + j])
    inv = 1.0 / n_samples
    for j in range(n_feat):
        mean_out[j] *= inv


def _compute_var(X, n_samples, n_feat, mean_, var_out):
    """
    Unbiased column variance (ddof=1) into var_out (float).
    var_out must be pre-allocated, length n_feat.
    mean_ must already be computed.
    """
    for j in range(n_feat):
        var_out[j] = 0.0
    for s in range(n_samples):
        base = s * n_feat
        for j in range(n_feat):
            d = float(X[base + j]) - mean_[j]
            var_out[j] += d * d
    inv = 1.0 / (n_samples - 1)
    for j in range(n_feat):
        var_out[j] *= inv


def _compute_scale(var_, n_feat, scale_out, with_std):
    """
    Compute per-feature scale factors into scale_out.
    with_std=True  -> scale = 1/std  (standard scaling)
    with_std=False -> scale = 1.0    (mean-only centering)
    Zero-variance features get scale=1.0 (no division by zero).
    scale_out must be pre-allocated, length n_feat.
    """
    for j in range(n_feat):
        if with_std:
            std = math.sqrt(var_[j]) if var_[j] > 0.0 else 1.0
            scale_out[j] = 1.0 / std
        else:
            scale_out[j] = 1.0


# ---------------------------------------------------------------------------
# StandardScaler class
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Standardise features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool  (default True)
        Subtract the mean from each feature.
    with_std  : bool  (default True)
        Scale each feature by its standard deviation.

    Attributes set after fit()
    --------------------------
    mean_       : array.array('f'), length n_features  (or None if with_mean=False)
    var_        : array.array('f'), length n_features  (or None if with_std=False)
    scale_      : array.array('f'), length n_features  (1/std, or 1.0 per feature)
    n_features_ : int
    n_samples_  : int

    Usage
    -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X, n_samples)   # batch
    x_scaled = scaler.transform_one(sample)          # single sample, on-device
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std  = with_std
        self.n_features_ = 0
        self.n_samples_  = 0
        self.mean_  = None
        self.var_   = None
        self.scale_ = None

    # ------------------------------------------------------------------
    def fit(self, X, n_samples):
        """
        Compute mean and std from X.

        Parameters
        ----------
        X        : array.array('h'), flat n_samples x n_features (int16)
        n_samples: int

        Returns
        -------
        self
        """
        n_feat = len(X) // n_samples
        self.n_features_ = n_feat
        self.n_samples_  = n_samples

        # Allocate all stat buffers up front
        self.mean_  = _alloc_f(n_feat)
        self.var_   = _alloc_f(n_feat)
        self.scale_ = _alloc_f(n_feat)

        _compute_mean(X, n_samples, n_feat, self.mean_)

        if self.with_std:
            _compute_var(X, n_samples, n_feat, self.mean_, self.var_)

        _compute_scale(self.var_, n_feat, self.scale_, self.with_std)

        return self

    # ------------------------------------------------------------------
    def transform(self, X, n_samples, out=None):
        """
        Apply scaling to X.

        Parameters
        ----------
        X        : array.array('h'), flat n_samples x n_features (int16)
        n_samples: int
        out      : optional pre-allocated array.array('f'),
                   length n_samples * n_features

        Returns
        -------
        out : array.array('f'), same shape as X
        """
        n_feat = self.n_features_
        if out is None:
            out = _alloc_f2(n_samples, n_feat)

        _apply_scaling(X, n_samples, n_feat,
                       self.mean_, self.scale_,
                       self.with_mean, out)
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, X, n_samples, out=None):
        """Fit then transform in one call."""
        self.fit(X, n_samples)
        return self.transform(X, n_samples, out)

    # ------------------------------------------------------------------
    def transform_one(self, sample, out=None):
        """
        Scale a single sample.

        Parameters
        ----------
        sample : array.array('h'), length n_features (int16)
        out    : optional pre-allocated array.array('f'), length n_features

        Returns
        -------
        out : array.array('f'), length n_features
        """
        n_feat = self.n_features_
        if out is None:
            out = _alloc_f(n_feat)

        for j in range(n_feat):
            v = float(sample[j])
            if self.with_mean:
                v -= self.mean_[j]
            out[j] = v * self.scale_[j]
        return out

    # ------------------------------------------------------------------
    def inverse_transform(self, X_scaled, n_samples, out=None):
        """
        Reverse the scaling: recover original (float) values.

        Parameters
        ----------
        X_scaled : array.array('f'), flat n_samples x n_features
        n_samples: int
        out      : optional pre-allocated array.array('f'),
                   length n_samples * n_features

        Returns
        -------
        out : array.array('f')
        """
        n_feat = self.n_features_
        if out is None:
            out = _alloc_f2(n_samples, n_feat)

        for s in range(n_samples):
            base = s * n_feat
            for j in range(n_feat):
                v = X_scaled[base + j] / self.scale_[j]
                if self.with_mean:
                    v += self.mean_[j]
                out[base + j] = v
        return out

    # ------------------------------------------------------------------
    def inverse_transform_one(self, sample_scaled, out=None):
        """
        Reverse scaling for a single scaled sample.

        Parameters
        ----------
        sample_scaled : array.array('f'), length n_features
        out           : optional pre-allocated array.array('f')

        Returns
        -------
        out : array.array('f'), length n_features
        """
        n_feat = self.n_features_
        if out is None:
            out = _alloc_f(n_feat)
        for j in range(n_feat):
            v = sample_scaled[j] / self.scale_[j]
            if self.with_mean:
                v += self.mean_[j]
            out[j] = v
        return out


# ---------------------------------------------------------------------------
# Internal transform (separated so it can be tested independently)
# ---------------------------------------------------------------------------

def _apply_scaling(X, n_samples, n_feat, mean_, scale_, with_mean, out):
    """
    Write scaled values into out (pre-allocated flat float array).
    X may be int16 or float.
    """
    for s in range(n_samples):
        base = s * n_feat
        for j in range(n_feat):
            v = float(X[base + j])
            if with_mean:
                v -= mean_[j]
            out[base + j] = v * scale_[j]


