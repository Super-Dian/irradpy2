import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def datetime_convert(datetime: np.ndarray):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def RuizArias2018CSD(ghi, dhi, zen, datetime: np.ndarray, plot_figure=None):
    """
    [1] Ruiz-Arias, José A.; Gueymard, Christian A. and Cebecauer, Tomas.
        2018. 'Direct Normal Irradiance Modeling : Evaluating the Impact on
        Accuracy of Worldwide Gridded Aerosol Databases', SolarPACES Conf.,
        Casablanca, Morocco, Oct. 2018.

    Coded by Jamie M. Bright 11/2018.
    Converted from Fortran code provided by Jose Ruiz-Arias 11/2018

    INPUTS
    ------
    ghi : array-like
        Global horizontal irradiance, column vector.
    dhi : array-like
        Diffuse horizontal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    datetime : np.ndarray
        Datetime input.
    plot_figure : optional
        If provided, plot the final clear-sky detection result.

    OUTPUTS
    -------
    csd : ndarray
        Clear-sky detection flag.
        1 = cloudy
        0 = clear
    """

    def _to_column_vector(arr, var_name):
        """
        Convert input to a MATLAB-style column vector.
        """
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            pass
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.T
        else:
            raise ValueError(f"var {var_name} must be a column vector")

        return arr

    def _hankel(first_col, last_row):
        """
        Build a Hankel matrix equivalent to MATLAB hankel(c, r).
        """
        first_col = np.asarray(first_col, dtype=float).reshape(-1)
        last_row = np.asarray(last_row, dtype=float).reshape(-1)

        values = np.concatenate([first_col, last_row[1:]])
        row_idx = np.arange(first_col.size)[:, None]
        col_idx = np.arange(last_row.size)[None, :]

        return values[row_idx + col_idx]

    # ---------------------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------------------
    ghi = _to_column_vector(ghi, "ghi")
    dhi = _to_column_vector(dhi, "dhi")
    zen = _to_column_vector(zen, "zen")
    Dayth, Year = datetime_convert(datetime)
    Dayth = Dayth.reshape(-1)
    Year = Year.reshape(-1)

    if len(set([len(ghi), len(dhi), len(zen), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    # ---------------------------------------------------------------------
    # Definitions needed
    # ---------------------------------------------------------------------
    dt = 1
    min_cs_points = 5
    max_iter = 10
    max_sza = 85
    csza = np.cos(np.deg2rad(zen))

    # find the unique days in the time series
    days = Year * 1000 + Dayth
    unique_days = np.unique(days)

    # pre allocate solar noon
    csza_at_noon = np.zeros_like(ghi)

    # loop through each unique day
    for i in range(len(unique_days)):
        # find the indices that correspond to this unique day
        idx = np.where(days == unique_days[i])[0]

        # find the minimum zenith this day, e.g. solar noon.
        csza_at_noon[idx, 0] = np.min(zen[idx, 0])

    # ---------------------------------------------------------------------
    # Global thresholds
    # ---------------------------------------------------------------------
    SC = 1365  # Solar constant, W/m2
    MAX_DIFFERENCE = 1e-5

    # 1. Normalized Total Shortwave Magnitude Test
    TEST1_LOW_SUN_THRES = np.cos(np.deg2rad(78.5))
    TEST1_MAX_INI_GHI = 1350
    TEST1_MIN_INI_GHI_WITH_LOW_SUN = 800
    TEST1_MIN_INI_GHI_WITH_HIGH_SUN = 900
    TEST1_INI_COEF_B = 1.2
    TEST1_UPPER_GHI_MARGIN = 150
    TEST1_LOWER_GHI_MARGIN = 150

    # 2. Maximum Diffuse Shortwave Test
    test2_max_dif = 150
    test2_coef_b = 0.5

    # 4. Normalized Diffuse Ratio Variability Test
    TEST4_INI_COEF_B = -0.8
    test4_max_sd = 0.0012

    min_cosz = np.cos(np.deg2rad(max_sza))

    # zen at noon.
    mu0 = np.nanmax(np.hstack([np.ones_like(csza) * 1e-4, csza]), axis=1, keepdims=True)
    mu0_at_noon = np.nanmax(np.hstack([np.ones_like(csza) * 1e-4, csza_at_noon]), axis=1, keepdims=True)
    glo = np.nanmax(np.hstack([np.ones_like(ghi) * 1e-4, ghi]), axis=1, keepdims=True)
    kd = np.nanmax(np.hstack([np.ones_like(ghi) * 1e-4, dhi / glo]), axis=1, keepdims=True)

    test0 = np.ones_like(ghi)
    test1 = np.ones_like(ghi)
    test2 = np.ones_like(ghi)
    test3 = np.ones_like(ghi)
    test4 = np.ones_like(ghi)

    # remove too small zeniths or missing data.
    test0[mu0 < min_cosz] = 0
    test1[np.isnan(ghi) | np.isnan(dhi)] = 0
    test2[np.isnan(ghi) | np.isnan(dhi)] = 0
    test3[np.isnan(ghi) | np.isnan(dhi)] = 0
    test4[np.isnan(ghi) | np.isnan(dhi)] = 0

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # FIRST LOOP THROUGHOUT DATA TO FLAG CLOUD OBSERVATIONS AND
    # FIND FIRST GUESS ESTIMATES FOR PARAMETERS OF TESTS 2 AND 4
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    test1_coef_b = TEST1_INI_COEF_B
    test4_coef_b = TEST4_INI_COEF_B

    # normalized diffuse ratio (it has to be computed here)
    norm_dif_ratio = kd / (mu0 ** test4_coef_b)

    # 1. Normalized Total Shortwave Magnitude Test
    norm_ghi = glo / (mu0 ** test1_coef_b)
    test1_max_allowed_ghi = np.ones_like(ghi) * TEST1_MAX_INI_GHI
    test1_min_allowed_ghi = np.ones_like(ghi) * TEST1_MIN_INI_GHI_WITH_LOW_SUN
    test1_min_allowed_ghi[mu0 < TEST1_LOW_SUN_THRES] = TEST1_MIN_INI_GHI_WITH_HIGH_SUN
    test1[(norm_ghi > test1_max_allowed_ghi) | (norm_ghi < test1_min_allowed_ghi)] = 0

    # 2. Maximum Diffuse Shortwave Test
    test2_dif = test2_max_dif * (mu0 ** test2_coef_b)
    test2[dhi > test2_dif] = 0

    # 3. Change in Magnitude with Time Test
    dif_mu0 = np.vstack([np.array([[np.nan]]), np.diff(mu0, axis=0)])
    dif_glo = np.vstack([np.array([[np.nan]]), np.diff(glo, axis=0)])
    Delta_Ftoa = SC * np.abs(dif_mu0 / (2.0 * dt))
    Delta_Gsfc = np.abs(dif_glo / (2.0 * dt))

    test3_max_allowed_var = Delta_Ftoa + 2.0 * mu0
    test3_min_allowed_var = Delta_Ftoa - (dt * (0.1 + mu0_at_noon) / mu0)
    test3[(Delta_Gsfc > test3_max_allowed_var) | (Delta_Gsfc < test3_min_allowed_var)] = 0

    # 4. Normalized Diffuse Ratio Variability Test
    test4_win_sz = 11
    test4_dat = _hankel(norm_dif_ratio.reshape(-1), np.concatenate(([norm_dif_ratio.reshape(-1)[-1]], np.full(test4_win_sz - 1, np.nan))))
    test4_running_mn = np.nanmean(test4_dat, axis=1, keepdims=True)
    test4_running_sd = np.sqrt(np.nansum((test4_dat - test4_running_mn) ** 2, axis=1, keepdims=True) / test4_win_sz)
    test4[test4_running_sd > test4_max_sd] = 0

    # define the csflag
    csflag = np.zeros_like(ghi)

    # apply the tests.
    csflag[(test0 + test2 + test3 + test4) == 4] = 1

    # apply the constraints.
    csflag[mu0 < min_cosz] = 0
    csflag[np.isnan(ghi) & np.isnan(dhi)] = 0
    n_cs_points = np.sum(csflag)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # HERE CSFLAG REPRESENTS THE BEST GUESS FROM FIRST PARAMETERISATION.
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # update clear-sky thresholds for test 1.
    log_x = np.log10(mu0[csflag == 1])
    log_y = np.log10(glo[csflag == 1])
    mean_x = np.nanmean(log_x)
    mean_y = np.nanmean(log_y)
    mean_xx = np.nanmean(log_x ** 2)
    mean_xy = np.nanmean(log_y * log_x)
    coef_b = (mean_xy - mean_x * mean_y) / (mean_xx - mean_x ** 2)
    coef_a = np.exp(mean_y - coef_b * mean_x)
    test1_max_allowed_ghi = coef_a + TEST1_UPPER_GHI_MARGIN
    test1_min_allowed_ghi = coef_a - TEST1_LOWER_GHI_MARGIN
    test1_coef_b = coef_b

    # update clear-sky thresholds for test 4.
    log_x = np.log10(mu0[csflag == 1])
    log_y = np.log10(kd[csflag == 1])
    mean_x = np.nanmean(log_x)
    mean_y = np.nanmean(log_y)
    mean_xx = np.nanmean(log_x ** 2)
    mean_xy = np.nanmean(log_y * log_x)
    coef_b = (mean_xy - mean_x * mean_y) / (mean_xx - mean_x ** 2)
    difference = np.abs(coef_b - test4_coef_b)
    test4_coef_b = coef_b

    # normalized diffuse ratio (it has to be computed here)
    norm_dif_ratio = kd / (mu0 ** test4_coef_b)

    n_iters = 0

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #             ITERATIVE OPTIMISATION PROCEDURE
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while (n_cs_points > min_cs_points) and (difference > MAX_DIFFERENCE) and (n_iters < max_iter):

        # 1. Normalized Total Shortwave Magnitude Test
        norm_ghi = glo / (mu0 ** test1_coef_b)
        test1[(norm_ghi > test1_max_allowed_ghi) | (norm_ghi < test1_min_allowed_ghi)] = 0

        # 4. Normalized Diffuse Ratio Variability Test
        test4_win_sz = 11
        test4_dat = _hankel(norm_dif_ratio.reshape(-1), np.concatenate(([norm_dif_ratio.reshape(-1)[-1]], np.full(test4_win_sz - 1, np.nan))))
        test4_running_mn = np.nanmean(test4_dat, axis=1, keepdims=True)
        test4_running_sd = np.sqrt(np.nansum((test4_dat - test4_running_mn) ** 2, axis=1, keepdims=True) / test4_win_sz)
        test4[test4_running_sd > test4_max_sd] = 0

        # apply the tests.
        csflag = np.zeros_like(ghi)
        csflag[(test0 + test2 + test3 + test4) == 4] = 1

        # apply the constraints.
        csflag[mu0 < min_cosz] = 0
        csflag[np.isnan(ghi) & np.isnan(dhi)] = 0

        n_cs_points = np.sum(csflag)

        # update clear-sky thresholds for test 1.
        log_x = np.log10(mu0[csflag == 1])
        log_y = np.log10(glo[csflag == 1])
        mean_x = np.nanmean(log_x)
        mean_y = np.nanmean(log_y)
        mean_xx = np.nanmean(log_x ** 2)
        mean_xy = np.nanmean(log_y * log_x)
        coef_b = (mean_xy - mean_x * mean_y) / (mean_xx - mean_x ** 2)
        coef_a = np.exp(mean_y - coef_b * mean_x)
        test1_max_allowed_ghi = coef_a + TEST1_UPPER_GHI_MARGIN
        test1_min_allowed_ghi = coef_a - TEST1_LOWER_GHI_MARGIN
        test1_coef_b = coef_b

        # update clear-sky thresholds for test4.
        log_x = np.log10(mu0[csflag == 1])
        log_y = np.log10(kd[csflag == 1])
        mean_x = np.nanmean(log_x)
        mean_y = np.nanmean(log_y)
        mean_xx = np.nanmean(log_x ** 2)
        mean_xy = np.nanmean(log_y * log_x)
        coef_b = (mean_xy - mean_x * mean_y) / (mean_xx - mean_x ** 2)
        difference = np.abs(coef_b - test4_coef_b)
        test4_coef_b = coef_b

        # normalized diffuse ratio (it has to be computed here)
        norm_dif_ratio = kd / (mu0 ** test4_coef_b)

        n_iters = n_iters + 1

        # if difference is smaller or equal than MAX_DIFFERENCE the
        # iteration is over and a last threshold of +-100 has to
        # be applied in test1
        if difference <= MAX_DIFFERENCE:
            test1_max_allowed_ghi = test1_max_allowed_ghi - 50
            test1_min_allowed_ghi = test1_min_allowed_ghi + 50

        # 1. Normalized Total Shortwave Magnitude Test
        norm_ghi = glo / (mu0 ** test1_coef_b)
        test1[(norm_ghi > test1_max_allowed_ghi) | (norm_ghi < test1_min_allowed_ghi)] = 0

        # apply the tests.
        csflag = np.zeros_like(ghi)
        csflag[(test0 + test2 + test3 + test4) == 4] = 1

        # apply the constraints.
        csflag[mu0 < min_cosz] = 0
        csflag[np.isnan(ghi) & np.isnan(dhi)] = 0

        n_cs_points = np.sum(csflag)

    csd = np.zeros_like(ghi)
    csd[csflag == 0] = 1

    # strange phenomenon that cannot have been intended is occasionally very
    # small irradiances being flagged as clear.

    # ---------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Ruiz Arias 2018 example", facecolor="w")
        plt.plot(ghi.reshape(-1))
        CSD = ghi.astype(float).copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD.reshape(-1), linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.gca().set_xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd