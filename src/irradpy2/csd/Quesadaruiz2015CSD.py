import numpy as np
import matplotlib.pyplot as plt


def Quesadaruiz2015CSD(ghi: np.ndarray, ghics: np.ndarray, plot_figure=None):
    """
    Quesada-Ruiz 2015 clear-sky detection method.

    References
    ----------
    [1] Quesada-Ruiz, Samuel Linares-Rodriguez, Alvaro Ruiz-Arias, José A. Pozo-Vázquez, David. 2015.
    An advanced ANN-based method to estimate hourly solar radiation from multi-spectral MSG imagery.
    Solar Energy, 115, 494-504.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance, column vector.
        Not necessary to be continuous so long as the ghics perfectly corresponds.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance, column vector.
    plot_figure : optional
        If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

    Outputs
    -------
    csd : np.ndarray
        Clear-sky detection.
        A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.

    Example
    -------
    csd = Quesadaruiz2015CSD(ghi, ghics, 1)
    """

    # safety checks
    ghi = np.array(ghi)
    ghics = np.array(ghics)

    if not (ghi.ndim == 1 or (ghi.ndim == 2 and ghi.shape[1] == 1)):
        ghi = ghi.T
        if not (ghi.ndim == 1 or (ghi.ndim == 2 and ghi.shape[1] == 1)):
            raise ValueError("var ghi must be a column vector")

    if not (ghics.ndim == 1 or (ghics.ndim == 2 and ghics.shape[1] == 1)):
        ghics = ghics.T
        if not (ghics.ndim == 1 or (ghics.ndim == 2 and ghics.shape[1] == 1)):
            raise ValueError("var ghics must be a column vector")

    if len(np.unique(np.array([len(ghi), len(ghics)]))) != 1:
        raise ValueError("vars must be equal in length")

    # calculations
    kc = ghi / ghics
    csd = np.zeros(ghi.shape)
    csd[kc < 0.8] = 1

    # figure example
    if plot_figure is not None:
        plt.figure(num="Quesada-Ruiz CSD example", facecolor="w")
        plt.plot(ghi)
        CSD = np.array(ghi, dtype=float)
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return np.asarray(csd, dtype=float).reshape(-1, 1)