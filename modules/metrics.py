import numpy as np
from scipy.spatial.distance import jensenshannon

def get_histogram(data, nbins=20, hist_range=None):
    """
    Estimate a normalized histogram (probability distribution).

    Parameters
    ----------
    data : 1D np.ndarray
        Raw data values, shape (n,).
    nbins : int
        Number of bins.
    hist_range : tuple or None
        (min, max) range for the histogram.

    Returns
    -------
    p : np.ndarray
        Probability histogram of shape (nbins,).
    bin_edges : np.ndarray
        Histogram bin edges.
    """
    hist, bin_edges = np.histogram(data, bins=nbins, range=hist_range)
    p = hist.astype(float)
    p /= p.sum() + 1e-12  # normalize

    return p, bin_edges

def get_js(arr1, arr2, nbins=20, hist_range=None):
    """
    Compute JS divergence per cell between histogram of raw samples and a reference distribution.
    """

    ns, ny, nx = arr1.shape

    js_map = np.zeros((ny, nx))

    # Loop over grid cells (cannot be avoided, each cell has its own distribution)
    for y in range(ny):
        for x in range(nx):
            data = arr1[:, y, x]    # shape (nsamples,)
            p, _ = get_histogram(data, nbins=nbins, hist_range=hist_range)
            data = arr2[:, y, x]
            q, _ = get_histogram(data, nbins=nbins, hist_range=hist_range)
            js_map[y, x] = jensenshannon(p, q)**2

    return js_map