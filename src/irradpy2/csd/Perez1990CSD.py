import numpy as np
import matplotlib.pyplot as plt


def Perez1990CSD(dhi, dni, zen, plot_figure=None):
    """
    Perez 1990 Clear-sky detection method [1]

    References
    ----------
    [1] Perez, Richard; Ineichen, Pierre; Seals, Robert; Michalsky, Joseph;
        Stewart, Ronald. 1990. Components from direct and global irradiance.
        Solar Energy. 44 (5), 271-289.

    Parameters
    ----------
    dhi : array-like
        Diffuse horizontal irradiance, column vector.
    dni : array-like
        Direct normal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    plot_figure : optional
        If provided, a figure is plotted illustrating the outcome of the CSD method.

    Returns
    -------
    csd : ndarray
        Clear-sky detection flag.
        A flag of 1 means that clouds were detected,
        whereas a flag of 0 means that the hour is clear.
    """

    # safety checks
    dhi = np.asarray(dhi, dtype=float)
    if not (dhi.ndim == 2 and dhi.shape[1] == 1):
        dhi = dhi.T
        if dhi.ndim == 1:
            dhi = dhi.reshape(-1, 1)
        if not (dhi.ndim == 2 and dhi.shape[1] == 1):
            raise ValueError("var dhi must be a column vector")

    dni = np.asarray(dni, dtype=float)
    if not (dni.ndim == 2 and dni.shape[1] == 1):
        dni = dni.T
        if dni.ndim == 1:
            dni = dni.reshape(-1, 1)
        if not (dni.ndim == 2 and dni.shape[1] == 1):
            raise ValueError("var dni must be a column vector")

    zen = np.asarray(zen, dtype=float)
    if not (zen.ndim == 2 and zen.shape[1] == 1):
        zen = zen.T
        if zen.ndim == 1:
            zen = zen.reshape(-1, 1)
        if not (zen.ndim == 2 and zen.shape[1] == 1):
            raise ValueError("var zen must be a column vector")

    if len(dhi) != len(dni) or len(dhi) != len(zen):
        raise ValueError("vars must be equal")

    # The sky's clearness, noted as e, is given according to eq. (1) in [1] as:
    k = 1.041
    e = ((dhi + dni) / dhi + k * np.deg2rad(zen) ** 3) / (1 + k * np.deg2rad(zen) ** 3)

    # apply the CSD test such that if e > 6.2, it is deemed to be clear.
    csd = np.zeros_like(dhi)
    csd[e < 6.2] = 1  # from table 1 in [1].

    # figure example
    if plot_figure is not None:
        plt.figure(num="Perez CSD example", facecolor="w")
        plt.clf()
        ghi = dhi + np.cos(np.deg2rad(zen)) * dni
        plt.plot(ghi)
        CSD = ghi.copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.gca().set_xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd