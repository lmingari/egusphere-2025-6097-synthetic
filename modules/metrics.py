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

def isotropic_spectrum_ensemble(fields, dx=1.0, dy=1.0, nbins=None):
    """
    fields: array (n_members, nx, ny)
    dx, dy: physical grid spacing in x,y (default 1.0)
    nbins: number of radial bins (default ~min(nx,ny)//2)
    returns: k_centers, E_mean(k), E_std(k)
      - E_mean(k): mean over members of each member's isotropic spectrum
      - E_std(k): std across members (for CI / shading)
      - n_modes(k): number of Fourier modes in each radial bin
    """
    n_members, nx, ny = fields.shape
    if nbins is None:
        nbins = min(nx, ny)//2

    fields_windowed = fields.copy()

    # FFT grid
    kx = np.fft.fftfreq(nx, d=dx)  # cycles per unit length
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    kmax = K.max()
    kbins = np.linspace(0.0, kmax, nbins+1)
    k_centers = 0.5*(kbins[:-1] + kbins[1:])

    # compute isotropic spectrum per member
    spectra = np.zeros((n_members, nbins))

    # Number of Fourier modes in each radial bin
    n_modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = (K >= kbins[i]) & (K < kbins[i+1])
        n_modes[i] = np.sum(mask)
            
    for m in range(n_members):
        F = np.fft.fftn(fields_windowed[m])
        P = np.abs(F)**2  # raw power
        # normalization so that Parseval roughly holds:
        # sum(field^2) ~= sum(P) / (nx*ny)**2  (depends on FFT conventions)
        P = P / (nx * ny)**2

        # radial binning
        for i in range(nbins):
            mask = (K >= kbins[i]) & (K < kbins[i+1])
            if np.any(mask):
                spectra[m, i] = P[mask].mean()
            else:
                spectra[m, i] = np.nan

    # average and std across members (ignore bins with nan)
    E_mean = np.nanmean(spectra, axis=0)
    E_std  = np.nanstd (spectra, axis=0)

    return k_centers, E_mean, E_std, n_modes