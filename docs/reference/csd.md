# `irradpy2.csd` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (23): `AliaMartinez2016CSD`, `Batlles1999CSD`, `BrightSun2020CSDc`, `BrightSun2020CSDs`, `Ellis2018CSD`, `Garcia2014CSD`, `Gueymard2013CSD`, `Ineichen2006CSD`, `Ineichen2009CSD`, `Ineichen2016CSD`, `Inman2015CSD`, `Larraneta2017CSD`, `Lefevre2013CSD`, `LongAckerman2000CSD`, `Perez1990CSD`, `Polo2009CSD`, `Quesadaruiz2015CSD`, `Reno2016CSD`, `RuizArias2018CSD`, `Shen2018CSD`, `Xie2013CSD`, `Zhang2018CSD`, `Zhao2018CSD`

---

### `AliaMartinez2016CSD(ghi, ghics, zen, datetime, plot_figure=None)`

Alia-Martinez 2016 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance [W m-2], column vector.
- `zen` (*np.ndarray*) — Zenith angle [deg], column vector.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

**References**

Alia-Martinez, M., Antonanzas, J., Urraca, R., Martinez-De-Pison, F. J.,
and Antonanzas-Torres, F. (2016).
Benchmark of algorithms for solar clear-sky detection.
Journal of Renewable and Sustainable Energy, 8(3).

Polo, J., Zarzalejo, L. F., Martín, L., Navarro, A. A., and Marchante, R. (2009).
Estimation of daily Linke turbidity factor by using global irradiance
measurements at solar noon. Solar Energy, 83(8), 1177-1185.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following parts are intentionally preserved:
- parameter names
- parameter order
- threshold values
- decision order
- intermediate logic

### `Batlles1999CSD(ghi, dhi, exth, zen, plot_figure=None)`

Batlles 1999 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance [W m-2], column vector.
- `dhi` (*np.ndarray*) — Diffuse horizontal irradiance [W m-2], column vector.
- `exth` (*np.ndarray*) — Horizontal projection of extraterrestrial irradiance [W m-2], column vector.
- `zen` (*np.ndarray*) — Zenith angle [degrees], column vector.
- `plot_figure` (*optional*) — If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 0 = clear 1 = cloudy

**References**

Batlles, F. J., Olmo, F. J., Tovar, J., & Alados-Arboledas, L. (1999).
Comparison of cloudless sky parametrizations of solar irradiance at various spanish midlatitude locations.
Theoretical and Applied Climatology, 66, 81-93.

Original MATLAB version coded by Jamie M. Bright, 11/2018.

**Notes**

Assumption: all data is in 1-min resolution and all vectors perfectly match each other in terms of time stamps.
This function is a strict Python translation of the MATLAB version.
The parameter names, threshold expressions, decision order, and intermediate logic are intentionally preserved.

### `BrightSun2020CSDc(zen, ghi, ghics, dhi, difcs, datetime, plot_figure=None)`

The Bright-Sun clear-sky detection methodology, 2020. Cloudless sky CSDc.  This function is a structural Python refactor of the original MATLAB implementation. Parameters, thresholds, criteria order, optimisation flow, and duration-filter logic are preserved.

**Parameters**

- `zen` (*array-like*) — Solar zenith angle [degrees], column vector.
- `ghi` (*array-like*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*array-like*) — Clear-sky global horizontal irradiance [W m-2], column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance [W m-2], column vector.
- `difcs` (*array-like*) — Clear-sky diffuse horizontal irradiance [W m-2], column vector.
- `datetime` (*array-like*) — Datetime input.
- `plot_figure` (*any, optional*) — If provided, diagnostic figures are plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection for cloudless sky. 1 means that clouds were detected. 0 means clear sky.
- `csd_tricomponent` (*np.ndarray*) — Clear-sky detection from the Reno tri-component optimisation.
- `ghics_optimised` (*np.ndarray*) — Optimised clear-sky GHI curve.
- `dnics_optimised` (*np.ndarray*) — Optimised clear-sky DNI curve.
- `difcs_optimised` (*np.ndarray*) — Optimised clear-sky diffuse (DIF) curve.

### `BrightSun2020CSDs(zen, ghi, ghics, dhi, difcs, datetime, plot_figure=None)`

Bright-Sun 2020 clear-sky detection methodology for cloudless sky CSDs.

**Inputs**

- `zen` (*np.ndarray*) — Zenith angle [deg], column vector.
- `ghi` (*np.ndarray*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance [W m-2], column vector.
- `dhi` (*np.ndarray*) — Diffuse horizontal irradiance [W m-2], column vector.
- `difcs` (*np.ndarray*) — Clear-sky diffuse horizontal irradiance [W m-2], column vector.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*optional*) — If provided, plot the intermediate and final clear-sky detection results.

**Outputs**

- `csd` (*np.ndarray*) — Final clear-sky detection flag for cloudless sky. 1 = clouds suspected 0 = clear
- `csd_tricomponent` (*np.ndarray*) — Tri-component clear-sky detection before duration filters.
- `ghics_optimised` (*np.ndarray*) — Optimised clear-sky GHI curve.
- `dnics_optimised` (*np.ndarray*) — Optimised clear-sky DNI curve.
- `difcs_optimised` (*np.ndarray*) — Optimised clear-sky DIF curve.

**References**

Alia-Martinez, M., Antonanzas, J., Urraca, R., Martinez-De-Pison, F. J., and Antonanzas-Torres, F.
(2016). Benchmark of algorithms for solar clear-sky detection. Journal of Renewable and Sustainable Energy, 8(3).

Ellis, 2018, PVLIB detect_ghics at commit f4e4ad3bbe335fa49bf300ce53784d69d719ca98.

Reno, M. J., and Hansen, C. W. (2016). Identification of periods of clear sky irradiance in time series of GHI measurements.
Renewable Energy, 90, 520-531.

Shen, Y., Wei, H., Zhu, T., Zhao, X., and Zhang, K. (2018). A Data-driven Clear Sky Model for Direct Normal Irradiance.
IOP Conference Series: Journal of Physics: Conference Series, 1072.

Inman, R. H., Edson, J. G., and Coimbra, C. F. M. (2015). Impact of local broadband turbidity estimation on forecasting of
clear sky direct normal irradiance. Solar Energy, 117, 125-138.

Gueymard, C. A., Bright, J. M., Lingfors, D., Habte, A., and Sengupta, M. (2019). A posteriori clear-sky identification
methods in solar irradiance time series: Review and preliminary validation using sky imagers.
Renewable and Sustainable Energy Reviews.

Larraneta, M., Reno, M. J., Lillo-Bravo, I., and Silva-Perez, M. A. (2017). Identifying periods of clear sky direct normal
irradiance. Renewable Energy, 113, 756-763.

Quesada-Ruiz, S., Linares-Rodriguez, A., Ruiz-Arias, J. A., and Pozo-Vazquez, D. (2015). An advanced ANN-based method to
estimate hourly solar radiation from multi-spectral MSG imagery. Solar Energy, 115, 494-504.

Zhang, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018). Modeling and Simulation of High Frequency Solar Irradiance.

Zhang, W., Kleiber, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018). A stochastic downscaling approach for generating
high-frequency solar irradiance scenarios. Solar Energy, 176, 370-379.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following parts are intentionally preserved:
- parameter names
- parameter order
- output order
- threshold values
- decision order
- optimisation logic
- duration filtering logic

### `Ellis2018CSD(ghi, ghics, plot_figure=None)`

Ellis 2018 clear-sky detection method.  This function is a direct structural Python translation of the MATLAB implementation. The original criteria thresholds, optimisation order, and output logic are preserved.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*array-like*) — Clear-sky global horizontal irradiance [W m-2], column vector; equal in length to ``ghi``.
- `plot_figure` (*any, optional*) — If provided, a diagnostic figure illustrating the outcome is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means that clouds were detected. 0 means clear sky.

### `Garcia2014CSD(ghi, dhi, zen, aod550, datetime, plot_figure=None)`

Garcia 2014 clear-sky detection method.  This function is a direct structural Python translation of the MATLAB implementation. The original thresholds, solar-noon procedure, variability test, and plotting logic are preserved.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance [W m-2], column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance [W m-2], column vector.
- `zen` (*array-like*) — Solar zenith angle [degrees], column vector.
- `aod550` (*array-like*) — Total aerosol extinction at 550 nm, column vector.
- `datetime` (*array-like*) — Datetime input; used to locate solar noon and build the day index.
- `plot_figure` (*any, optional*) — If provided, a diagnostic figure is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 means that clouds were detected. 0 means clear sky.

**Notes**

Assumes 1-minute data resolution and perfectly time-matched input vectors.

### `Gueymard2013CSD(dni, dnics, zen, plot_figure=None)`

Gueymard 2013 clear-sky detection method.

**Parameters**

- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `dnics` (*array-like*) — Clear-sky direct normal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `plot_figure` (*any, optional*) — If provided, a diagnostic figure is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 = cloudy 0 = clear

**References**

C.A. Gueymard. 2013.
Aerosol turbidity derivation from broadband irradiance measurements:
methodological advances and uncertainty analysis.
ASES Solar 2013 Conf., Baltimore, MD (2013)

### `Ineichen2006CSD(exth, dni, zen, plot_figure=None)`

Ineichen 2006 clear-sky detection model.  This is a structural Python refactor of the original MATLAB implementation. Parameters, thresholds, criteria order, and plotting logic are preserved.

**Parameters**

- `exth` (*array-like*) — Horizontal projection of extraterrestrial irradiance, column vector.
- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `plot_figure` (*any, optional*) — If provided (not None), plot the detection result.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 0 means clear sky, 1 means clouds detected.

### `Ineichen2009CSD(ghi, exth, zen, plot_figure=None)`

Ineichen 2009 Clear Sky Detection model.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `exth` (*array-like*) — Horizontal projection of extraterrestrial irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `plot_figure` (*any, optional*) — If not None, plot the detection result.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means clouds were detected, 0 means clear sky.

### `Ineichen2016CSD(ghi, dni, dhi, zen, exth, aod550, plot_figure=None)`

Ineichen 2016 Clear Sky Detection model.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `exth` (*array-like*) — Horizontal projection of extraterrestrial irradiance, column vector.
- `aod550` (*array-like*) — Total aerosol extinction at 550 nm, column vector.
- `plot_figure` (*any, optional*) — If not None, plot the detection result.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means clouds were detected, 0 means clear sky.

### `Inman2015CSD(ghi, ghics, dni, dnics, plot_figure=None)`

Inman 2015 Clear Sky Detection model.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `ghics` (*array-like*) — Clear-sky global horizontal irradiance, column vector.
- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `dnics` (*array-like*) — Clear-sky direct normal irradiance, column vector.
- `plot_figure` (*any, optional*) — If provided, a figure illustrating the outcome is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means that clouds were detected. 0 means clear sky.

### `Larraneta2017CSD(dni, dnics, zen, datetime, plot_figure=None)`

Larraneta 2017 Clear Sky Detection model.

**Parameters**

- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `dnics` (*array-like*) — Clear-sky direct normal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*any, optional*) — If defined, a figure illustrating the CSD output is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means clouds were detected. 0 means clear sky.

### `Lefevre2013CSD(ghi, dhi, exth, zen, plot_figure=None)`

Lefevre 2013 Clear Sky Detection model.  Structural Python refactor of the original MATLAB implementation. Parameters, thresholds, logic order, and output meaning are preserved.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance, column vector.
- `exth` (*array-like*) — Horizontal projection of extraterrestrial irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `plot_figure` (*any, optional*) — If defined, a figure illustrating the result is plotted.

**Returns**

- `csd` (*np.ndarray*) — Clear-sky detection flag as a column vector. 1 means clouds were detected. 0 means clear sky.

### `LongAckerman2000CSD(ghi, dhi, zen, datetime, plot_figure=None)`

Long and Ackerman (2000) clear-sky detection model.  This is a structural Python translation of the original MATLAB code. Parameters, thresholds, decision logic, and output meaning are preserved.

**Parameters**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*any, optional*) — If provided, a figure illustrating the result is plotted.

**Returns**

np.ndarray     Clear-sky detection flag as a column vector.     1 means clouds were detected.     0 means clear sky.

### `Perez1990CSD(dhi, dni, zen, plot_figure=None)`

Perez 1990 Clear-sky detection method [1]

**Parameters**

- `dhi` (*array-like*) — Diffuse horizontal irradiance, column vector.
- `dni` (*array-like*) — Direct normal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `plot_figure` (*optional*) — If provided, a figure is plotted illustrating the outcome of the CSD method.

**Returns**

- `csd` (*ndarray*) — Clear-sky detection flag. A flag of 1 means that clouds were detected, whereas a flag of 0 means that the hour is clear.

**References**

[1] Perez, Richard; Ineichen, Pierre; Seals, Robert; Michalsky, Joseph;
    Stewart, Ronald. 1990. Components from direct and global irradiance.
    Solar Energy. 44 (5), 271-289.

### `Polo2009CSD(ghi, ghics, zen, datetime, plot_figure=None)`

Polo 2009 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance, column vector. Not necessary to be continuous so long as the ghics perfectly corresponds.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance, column vector.
- `zen` (*np.ndarray*) — Zenith angle in degrees. Corresponding to all inputs.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*optional*) — If provided, a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection. A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.

**References**

J. Polo, L. F. Zarzalejo, L. Martín, A. A. Navarro, and R. Marchante,
‘Good CSDs methodology, not a good CSDc, measurements at solar noon’,
Sol. Energy, vol. 83, no. 8, pp. 1177–1185, 2009.

### `Quesadaruiz2015CSD(ghi, ghics, plot_figure=None)`

Quesada-Ruiz 2015 clear-sky detection method.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance, column vector. Not necessary to be continuous so long as the ghics perfectly corresponds.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance, column vector.
- `plot_figure` (*optional*) — If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection. A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.

**References**

[1] Quesada-Ruiz, Samuel Linares-Rodriguez, Alvaro Ruiz-Arias, José A. Pozo-Vázquez, David. 2015.
An advanced ANN-based method to estimate hourly solar radiation from multi-spectral MSG imagery.
Solar Energy, 115, 494-504.

**Example**

```python
csd = Quesadaruiz2015CSD(ghi, ghics, 1)
```

### `Reno2016CSD(ghi, ghics, plot_figure=None)`

Reno 2016 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance, column vector. Not necessary to be continuous so long as the ghics perfectly corresponds.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance, column vector.
- `plot_figure` (*optional*) — If this variable is defined by any class, then a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection. A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.

**References**

Reno, M. J. and Hansen, C. W. (2016).
Identification of periods of clear sky irradiance in time series of GHI measurements.
Renewable Energy, 90, 520-531.

**Example**

```python
csd = Reno2016CSD(ghi, ghics, 1)
```

### `RuizArias2018CSD(ghi, dhi, zen, datetime, plot_figure=None)`

[1] Ruiz-Arias, José A.; Gueymard, Christian A. and Cebecauer, Tomas.     2018. 'Direct Normal Irradiance Modeling : Evaluating the Impact on     Accuracy of Worldwide Gridded Aerosol Databases', SolarPACES Conf.,     Casablanca, Morocco, Oct. 2018.  Coded by Jamie M. Bright 11/2018. Converted from Fortran code provided by Jose Ruiz-Arias 11/2018

**Inputs**

- `ghi` (*array-like*) — Global horizontal irradiance, column vector.
- `dhi` (*array-like*) — Diffuse horizontal irradiance, column vector.
- `zen` (*array-like*) — Zenith angle in degrees, column vector.
- `datetime` (*np.ndarray*) — Datetime input.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

### `Shen2018CSD(dni, dnics, aod550, plot_figure=None)`

Shen 2018 clear-sky detection model.

**Inputs**

- `dni` (*np.ndarray*) — Direct normal irradiance [W m-2], column vector.
- `dnics` (*np.ndarray*) — Clear-sky direct normal irradiance [W m-2], column vector.
- `aod550` (*np.ndarray*) — Total aerosol extinction at 550 nm, column vector.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

**References**

Shen, Y., Wei, H., Zhu, T., Zhao, X., and Zhang, K. (2018).
A Data-driven Clear Sky Model for Direct Normal Irradiance.
IOP Conference Series: Journal of Physics: Conference Series, 1072.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following parts are intentionally preserved:
- parameter meaning
- threshold values
- decision order
- intermediate logic
- output definition

### `Xie2013CSD(ghi, ghics, dni, dnics, zen, exth, plot_figure=None)`

Xie 2013 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance [W m-2], column vector.
- `dni` (*np.ndarray*) — Direct normal irradiance [W m-2], column vector.
- `dnics` (*np.ndarray*) — Clear-sky direct normal irradiance [W m-2], column vector.
- `zen` (*np.ndarray*) — Zenith angle [deg], column vector.
- `exth` (*np.ndarray*) — Horizontal projection of extraterrestrial irradiance, column vector.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

**References**

Xie, Y., and Liu, Y. (2013).
A new approach for simultaneously retrieving cloud albedo and cloud fraction from surface-based shortwave radiation measurements.
Environmental Research Letters, 8(044023), 1-9.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following parts are intentionally preserved:
- parameter names
- parameter order
- threshold values
- decision order
- intermediate logic
- output definition

### `Zhang2018CSD(ghi, ghics, plot_figure=None)`

Zhang 2018 clear-sky detection model.

**Inputs**

- `ghi` (*np.ndarray*) — Global horizontal irradiance [W m-2], column vector.
- `ghics` (*np.ndarray*) — Clear-sky global horizontal irradiance [W m-2], column vector.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

**References**

Zhang, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018).
Modeling and Simulation of High Frequency Solar Irradiance.
Preprint to Journal of Photovoltaics.

Zhang, W., Kleiber, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018).
A stochastic downscaling approach for generating high-frequency solar irradiance scenarios.
Solar Energy, 176, 370-379.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following parts are intentionally preserved:
- parameter names
- parameter order
- threshold values
- decision order
- intermediate logic
- output definition

### `Zhao2018CSD(dni, dnics, plot_figure=None)`

Zhao 2018 clear-sky detection model.

**Inputs**

- `dni` (*np.ndarray*) — Direct normal irradiance [W m-2], column vector.
- `dnics` (*np.ndarray*) — Clear-sky direct normal irradiance [W m-2], column vector.
- `plot_figure` (*optional*) — If provided, plot the final clear-sky detection result.

**Outputs**

- `csd` (*np.ndarray*) — Clear-sky detection flag. 1 = cloudy 0 = clear

**References**

Zhao, X., Wei, H., Shen, Y., and Zhang, K. (2018).
Real-time Clear-sky Model and Cloud Cover for Direct Normal Irradiance Prediction.
IOP Conference Series: Journal of Physics: Conference Series, 1072.

**Notes**

This function is a strict Python translation of the MATLAB version.
The following are intentionally preserved:
- parameter names
- parameter order
- threshold values
- algorithm flow
- output definition

