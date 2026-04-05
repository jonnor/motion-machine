"""
PCA - scikit-learn style API for MicroPython
Input: array.array with typecode 'h' (int16)
Internal computation: float
"""

import math
import array


# ---------------------------------------------------------------------------
# Low-level array helpers
# ---------------------------------------------------------------------------

def _alloc_f(n):
    """Allocate float array of length n, zero-filled."""
    return array.array('f', (0.0 for _ in range(n)))

def _alloc_f2(rows, cols):
    """Allocate flat float array for a rows x cols matrix."""
    return array.array('f', (0.0 for _ in range(rows * cols)))

def _idx(row, col, ncols):
    return row * ncols + col

def _dot_ff(a, b, n):
    """Dot product of two float arrays of length n."""
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

def _dot_fi(a_float, b_offset, b_flat, n):
    """Dot of float array a with row starting at b_offset in flat array."""
    s = 0.0
    for i in range(n):
        s += a_float[i] * b_flat[b_offset + i]
    return s

def _norm_f(v, n):
    s = 0.0
    for i in range(n):
        s += v[i] * v[i]
    return math.sqrt(s)

def _normalize_inplace(v, n):
    """Normalize float array v in-place."""
    mag = _norm_f(v, n)
    if mag < 1e-12:
        return
    inv = 1.0 / mag
    for i in range(n):
        v[i] *= inv

def _copy_f(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


# ---------------------------------------------------------------------------
# Matrix operations on flat float arrays
# ---------------------------------------------------------------------------

def _mat_vec(M, v, out, rows, cols):
    """out = M @ v,  M is flat (rows x cols), v length cols, out length rows."""
    for i in range(rows):
        s = 0.0
        base = i * cols
        for j in range(cols):
            s += M[base + j] * v[j]
        out[i] = s

def _covariance_matrix(X_flat, n_samples, n_feat, cov_out):
    """
    Compute covariance matrix into cov_out (flat n_feat x n_feat).
    X_flat is already mean-centered (flat n_samples x n_feat).
    """
    inv = 1.0 / (n_samples - 1)
    for i in range(n_feat):
        for j in range(i, n_feat):
            s = 0.0
            for k in range(n_samples):
                s += X_flat[_idx(k, i, n_feat)] * X_flat[_idx(k, j, n_feat)]
            val = s * inv
            cov_out[_idx(i, j, n_feat)] = val
            cov_out[_idx(j, i, n_feat)] = val  # symmetric

def _deflate_inplace(M, eigenvalue, eigvec, n, tmp):
    """
    Hotelling deflation: M -= eigenvalue * outer(eigvec, eigvec)
    tmp: scratch float array of length n (pre-allocated).
    """
    for i in range(n):
        for j in range(n):
            M[_idx(i, j, n)] -= eigenvalue * eigvec[i] * eigvec[j]


# ---------------------------------------------------------------------------
# Power iteration
# ---------------------------------------------------------------------------

def _power_iteration(M, n, n_iter, vec_out, tmp):
    """
    Find dominant eigenvector of symmetric matrix M (flat n x n).
    vec_out, tmp: pre-allocated float arrays of length n.
    Returns eigenvalue.
    """
    # initialise to uniform
    init = 1.0 / math.sqrt(n)
    for i in range(n):
        vec_out[i] = init

    for _ in range(n_iter):
        _mat_vec(M, vec_out, tmp, n, n)
        _copy_f(tmp, vec_out, n)
        _normalize_inplace(vec_out, n)

    # Rayleigh quotient for eigenvalue
    _mat_vec(M, vec_out, tmp, n, n)
    return _dot_ff(vec_out, tmp, n)


# ---------------------------------------------------------------------------
# PCA class
# ---------------------------------------------------------------------------

class PCA:
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    n_iter : int
        Power iteration steps per component (default 100).

    Attributes (set after fit)
    --------------------------
    components_      : flat array.array('f'), shape n_components x n_features
    explained_var_   : array.array('f'), length n_components
    explained_ratio_ : array.array('f'), length n_components
    mean_            : array.array('f'), length n_features
    n_features_      : int
    """

    def __init__(self, n_components=2, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        # set by fit()
        self.n_features_ = 0
        self.mean_            = None
        self.components_      = None
        self.explained_var_   = None
        self.explained_ratio_ = None

    # ------------------------------------------------------------------
    def fit(self, X, n_samples):
        """
        Fit PCA on dataset X.

        Parameters
        ----------
        X        : array.array('h') or flat sequence, shape n_samples x n_features
                   Typecode 'h' (int16) is accepted; converted to float internally.
        n_samples: int  — number of rows in X
        """
        n_feat = len(X) // n_samples
        self.n_features_ = n_feat
        nc = self.n_components

        # --- allocate all buffers up front ---
        self.mean_            = _alloc_f(n_feat)
        self.components_      = _alloc_f2(nc, n_feat)
        self.explained_var_   = _alloc_f(nc)
        self.explained_ratio_ = _alloc_f(nc)

        X_centered = _alloc_f2(n_samples, n_feat)  # working copy
        cov        = _alloc_f2(n_feat, n_feat)
        vec        = _alloc_f(n_feat)               # current eigenvector
        tmp        = _alloc_f(n_feat)               # scratch

        # --- compute mean ---
        _compute_mean(X, n_samples, n_feat, self.mean_)

        # --- mean-center into float buffer ---
        _center(X, n_samples, n_feat, self.mean_, X_centered)

        # --- covariance matrix ---
        _covariance_matrix(X_centered, n_samples, n_feat, cov)

        # --- extract components via power iteration + deflation ---
        total_var = 0.0
        for c in range(nc):
            eigenvalue = _power_iteration(cov, n_feat, self.n_iter, vec, tmp)
            # store component
            base = c * n_feat
            for i in range(n_feat):
                self.components_[base + i] = vec[i]
            self.explained_var_[c] = eigenvalue
            total_var += eigenvalue
            _deflate_inplace(cov, eigenvalue, vec, n_feat, tmp)

        # --- explained variance ratio ---
        inv_total = 1.0 / (total_var + 1e-12)
        for c in range(nc):
            self.explained_ratio_[c] = self.explained_var_[c] * inv_total

        return self

    # ------------------------------------------------------------------
    def transform(self, X, n_samples, out=None):
        """
        Project X onto principal components.

        Parameters
        ----------
        X        : array.array('h'), flat n_samples x n_features
        n_samples: int
        out      : optional pre-allocated array.array('f') of length
                   n_samples * n_components. Allocated if not provided.

        Returns
        -------
        out : array.array('f'), flat n_samples x n_components
        """
        n_feat = self.n_features_
        nc = self.n_components

        if out is None:
            out = _alloc_f(n_samples * nc)

        tmp = _alloc_f(n_feat)  # scratch for one centered sample

        for s in range(n_samples):
            # center sample into tmp
            base_in = s * n_feat
            for i in range(n_feat):
                tmp[i] = float(X[base_in + i]) - self.mean_[i]
            # project onto each component
            base_out = s * nc
            for c in range(nc):
                base_comp = c * n_feat
                out[base_out + c] = _dot_fi(tmp, base_comp, self.components_, n_feat)

        return out

    # ------------------------------------------------------------------
    def fit_transform(self, X, n_samples, out=None):
        """Fit then transform in one call."""
        self.fit(X, n_samples)
        return self.transform(X, n_samples, out)

    # ------------------------------------------------------------------
    def transform_one(self, sample, out=None):
        """
        Project a single sample (array.array 'h', length n_features).

        Returns array.array('f') of length n_components.
        """
        n_feat = self.n_features_
        nc = self.n_components
        if out is None:
            out = _alloc_f(nc)
        tmp = _alloc_f(n_feat)
        for i in range(n_feat):
            tmp[i] = float(sample[i]) - self.mean_[i]
        for c in range(nc):
            base_comp = c * n_feat
            out[c] = _dot_fi(tmp, base_comp, self.components_, n_feat)
        return out


# ---------------------------------------------------------------------------
# Internal helpers used by fit()
# ---------------------------------------------------------------------------

def _compute_mean(X, n_samples, n_feat, mean_out):
    """Compute column means of int16 flat array X into float mean_out."""
    for j in range(n_feat):
        mean_out[j] = 0.0
    for s in range(n_samples):
        base = s * n_feat
        for j in range(n_feat):
            mean_out[j] += float(X[base + j])
    inv = 1.0 / n_samples
    for j in range(n_feat):
        mean_out[j] *= inv

def _center(X, n_samples, n_feat, mean_, out):
    """Subtract mean from X (int16) into out (float), both flat."""
    for s in range(n_samples):
        base = s * n_feat
        for j in range(n_feat):
            out[base + j] = float(X[base + j]) - mean_[j]



