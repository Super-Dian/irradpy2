# `irradpy2.clear_sky` API reference

One entry per public function. Function names are listed in the module
overview; parameter semantics follow the source docstrings.

Reference lists are reproduced verbatim from the source docstrings.
They attribute the published formulation each model implements;
they are implementation provenance, not verified bibliographic records.

Functions (62): `abcgs`, `adnot`, `ashrae_1985`, `ashrae_2005`, `atwaterball`, `atwaterball2`, `badescu`, `bashahu`, `bclsm`, `biga`, `bird`, `calinoiu`, `campbell`, `capderou`, `cem`, `cls`, `cpcr2`, `csr`, `dai`, `dogniaux`, `dpp`, `elmghouchi`, `esra`, `furich`, `heliosat`, `heliosat2`, `hlj`, `hoyt`, `hs`, `ideriah`, `ineichen_2008`, `ineichen_2018`, `ineichenperez`, `iqbal_a`, `iqbal_b`, `iqbal_c`, `janjai`, `josefsson`, `kasm`, `kasten`, `king`, `kumar`, `mac1`, `mac2`, `metstat`, `mmac`, `modified_iqbalc`, `mrmv5`, `mrmv6`, `paltridge`, `paulescu`, `perrin`, `pr`, `psim`, `rest2v5`, `rsc`, `schulze`, `sharma`, `tj`, `watt`, `yang`, `yangwalsh`

---

### `abcgs(sza)`

ABCGS 1979 clear-sky model.

**Inputs**

- `sza` (*array-like*) — Solar zenith angle [radians]

**Outputs**

Returns [EghABCGS, EbnABCGS, EdhABCGS]:

- `EghABCGS` — Global horizontal irradiance [W m-2]
- `EbnABCGS` — Direct normal irradiance [W m-2]
- `EdhABCGS` — Diffuse horizontal irradiance [W m-2]

**References**

Schulze, R. E. (1976).
A physically based method of estimating solar radiation from suncards.

Adnot, J., Bourges, B., Campana, D., & Gicquel, R. (1979).
Utilisation des courbes de frequence cumulees pour le calcul des
installations solaires. In R. Lestienne (Ed.), Analyse statistique
des processus meteorologiques appliquee a l'energie solaire.
CNRS, Paris.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models
using routine hourly global irradiance measurements in Romania.

### `adnot(zsa)`

Original Adnot clear-sky model.

**Parameters**

- `zsa` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `EghAdnot` (*np.ndarray*) — Global horizontal irradiance [W m-2]

**References**

Adnot, J., Bourges, B., Campana, D., & Gicquel, R. (1979).
Utilisation de courbes de fréquences cumulées d'irradiation solaire globale
pour le calcul des installations solaires.

Barbieri, F., Rifflart, C., Vo, B. T., Rajakaruna, S., & Ghosh, A. (2016, July).
A comparative study of clear-sky irradiance models for Western Australia.
In 2016 IEEE Power and Energy Society General Meeting (PESGM) (pp. 1-5). IEEE.

### `ashrae_1985(sza, datetime)`

Original ASHRAE clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Month automatically.

**Returns**

- `[EghASHRAE, EbnASHRAE, EdhASHRAE]` (*list of np.ndarray*) — EghASHRAE : Global horizontal irradiance [W m-2] EbnASHRAE : Direct normal irradiance [W m-2] EdhASHRAE : Diffuse horizontal irradiance [W m-2]

**References**

Handbook, A. F. (1985).
American society of heating, refrigerating and air-conditioning engineers.
Inc.: Atlanta, GA, USA.

El Mghouchi, Y., Ajzoul, T., Taoukil, D., & El Bouardi, A. (2016).
The most suitable prediction model of the solar intensity, on horizontal plane,
at various weather conditions in a specified location in Morocco.
Renewable and Sustainable Energy Reviews, 54, 84-98.

### `ashrae_2005(sza, datetime)`

ASHRAE 2005 clear-sky model.

**Inputs**

- `sza` (*array-like*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to obtain the month automatically.

**Outputs**

Returns [EghASHRAE, EbnASHRAE, EdhASHRAE]:

- `EghASHRAE` — Global horizontal irradiance [W m-2]
- `EbnASHRAE` — Direct normal irradiance [W m-2]
- `EdhASHRAE` — Diffuse horizontal irradiance [W m-2]

**References**

ASHRAE. (2005).
ASHRAE Handbook—Fundamentals.
American Society of Heating, Refrigerating and Air-Conditioning
Engineers, Atlanta, GA, USA.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models
using routine hourly global irradiance measurements in Romania.

### `atwaterball(sza, datetime, press, wv)`

Original Atwater and Ball_1 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `Eghatwater1` (*np.ndarray*) — Global horizontal irradiance [W m-2]

**References**

Atwater, M. A., & Ball, J. T. (1981). A surface solar radiation model for cloudy atmospheres.
Monthly Weather Review, 109(4), 878-888.

### `atwaterball2(sza, datetime, press, wv)`

Original Atwater and Ball_1 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `Eghatwater1` (*np.ndarray*) — Global horizontal irradiance [W m-2]

**References**

Atwater, M. A., & Ball, J. T. (1981). A surface solar radiation model for cloudy atmospheres.
Monthly Weather Review, 109(4), 878-888.

### `badescu(sza, datetime, press, wv, ozone)`

Original Badescu clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]

**Returns**

- `[EghBadescu, EbnBadescu, EdhBadescu]` (*list of np.ndarray*) — EghBadescu : Global horizontal irradiance [W m-2] EbnBadescu : Direct normal irradiance [W m-2] EdhBadescu : Diffuse horizontal irradiance [W m-2]

**References**

Badescu, V. (2008). Use of sunshine number for solar irradiance time series generation.
In Modeling Solar Radiation at the Earth’s Surface (pp. 327-355). Springer, Berlin, Heidelberg.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `bashahu(sza, datetime, press, ang_alpha, ang_beta, wv)`

Original Bashahu clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `ang_alpha` (*np.ndarray*) — Angstrom exponent [dimensionless]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient [dimensionless]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghBashahu, EbnBashahu, EdhBashahu]` (*list of np.ndarray*) — EghBashahu : Global horizontal irradiance [W m-2] EbnBashahu : Direct normal irradiance [W m-2] EdhBashahu : Diffuse horizontal irradiance [W m-2]

**References**

Bashahu, M., & Laplaze, D. (1994). An atmospheric model for computing solar radiation.
Renewable Energy, 4(4), 455-458.

### `bclsm(sza, datetime, wv)`

BCLSM 1979 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]

**Returns**

- `[EghBCLSM, EbnBCLSM, EdhBCLSM]` (*list of np.ndarray*) — EghBCLSM : Global horizontal irradiance [W m-2] EbnBCLSM : Direct normal irradiance [W m-2] EdhBCLSM : Diffuse horizontal irradiance [W m-2]

**References**

Barbaro, S. (1979).
An atmospheric model for computing direct and diffuse solar radiation.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models
using routine hourly global irradiance measurements in Romania.

**Notes**

Diffuse horizontal irradiance is calculated from
EghBCLSM - EbnBCLSM * cos(sza).

### `biga(sza)`

Original Biga clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `[EghBiga, EbnBiga, EdhBiga]` (*list of np.ndarray*) — EghBiga : Global horizontal irradiance [W m-2] EbnBiga : Direct normal irradiance [W m-2] EdhBiga : Diffuse horizontal irradiance [W m-2]

**References**

Schulze, R. E. (1976).
A physically based method of estimating solar radiation from suncards.
Agricultural Meteorology, 16(1), 85-101.

### `bird(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, wv)`

Original Bird clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghbird, Ebnbird, Edhbird]` (*list of np.ndarray*) — Eghbird : Global horizontal irradiance [W m-2] Ebnbird : Direct normal irradiance [W m-2] Edhbird : Diffuse horizontal irradiance [W m-2]

**References**

Bird, R. E., & Hulstrom, R. L. (1981). Simplified clear sky model for direct and diffuse insolation on horizontal surfaces
(No. SERI/TR-642-761). Solar Energy Research Inst., Golden, CO (USA).

### `calinoiu(sza, datetime, ang_beta, ozone, NO2, wv)`

Original Calinoiu clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `NO2` (*np.ndarray*) — Total columnar amount of nitrogen dioxide [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghCalinoiu, EbnCalinoiu, EdhCalinoiu]` (*list of np.ndarray*) — EghCalinoiu : Global horizontal irradiance [W m-2] EbnCalinoiu : Direct normal irradiance [W m-2] EdhCalinoiu : Diffuse horizontal irradiance [W m-2]

**References**

Calinoiu, D., Stefu, N., Boata, R., Blaga, R., Pop, N., Paulescu, E., ... & Paulescu, M. (2018).
Parametric modeling: A simple and versatile route to solar irradiance.
Energy Conversion and Management, 164, 175-187.

### `campbell(sza, datetime, press)`

Original Campbell clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]

**Returns**

- `[EghCampbell, EbnCampbell, EdhCampbell]` (*list of np.ndarray*) — EghCampbell : Global horizontal irradiance [W m-2] EbnCampbell : Direct normal irradiance [W m-2] EdhCampbell : Diffuse horizontal irradiance [W m-2]

**References**

Campbell, G. S., & Norman, J. M. (1998).
An introduction to environmental biophysics. An introduction to environmental biophysics., (Ed. 2).

### `capderou(sza, datetime, latitude, altitude, press)`

Original Capderou clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `latitude` (*np.ndarray*) — Location latitude [-90, 90]
- `altitude` (*np.ndarray*) — Location altitude [m]
- `press` (*np.ndarray*) — Local barometric pressure [mb]

**Returns**

- `[EghCapderou, EbnCapderou, EdhCapderou]` (*list of np.ndarray*) — EghCapderou : Global horizontal irradiance [W m-2] EbnCapderou : Direct normal irradiance [W m-2] EdhCapderou : Diffuse horizontal irradiance [W m-2]

**References**

Capderou, M. (1987). Theoretical and experimental models solar atlas of Algeria (in French) Tome 1 and 2.
Algeria: University Publications Office.

Marif, Y., Chiba, Y., Belhadj, M. M., Zerrouki, M., & Benhammou, M. (2018).
A clear sky irradiation assessment using a modified Algerian solar atlas model in Adrar city.
Energy Reports, 4, 84-90.

### `cem(sza, datetime, press, albedo, broadbandaod, wv)`

Original CEM clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `broadbandaod` (*np.ndarray*) — Broadband aerosol optical depth [dimensionless]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghCEM, EbnCEM, EdhCEM]` (*list of np.ndarray*) — EghCEM : Global horizontal irradiance [W m-2] EbnCEM : Direct normal irradiance [W m-2] EdhCEM : Diffuse horizontal irradiance [W m-2]

**References**

Atwater, M. A., & Ball, J. T. (1978). A numerical solar radiation model based on standard meteorological observations.
Solar Energy, 21(3), 163-170.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `cls(sza, datetime, press, wv, albedo)`

Original CLS clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]
- `albedo` (*np.ndarray*) — Surface/ground albedo

**Returns**

- `[EghCLS, EbnCLS, EdhCLS]` (*list of np.ndarray*) — EghCLS : Global horizontal irradiance [W m-2] EbnCLS : Direct normal irradiance [W m-2] EdhCLS : Diffuse horizontal irradiance [W m-2]

**References**

Suckling, P. W., & Hay, J. E. (1976). Modelling direct, diffuse, and total solar radiation for cloudless days.
Atmosphere, 14(4), 298-308.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `cpcr2(sza, datetime, pressure, wv, ozone, ang_beta, ang_alpha, albedo)`

CPCR2 1989 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ozone` (*np.ndarray*) — Total ozone amount [atm-cm]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghCPCR2, EbnCPCR2, EdhCPCR2]` (*list of np.ndarray*) — EghCPCR2 : Global horizontal irradiance [W m-2] EbnCPCR2 : Direct normal irradiance [W m-2] EdhCPCR2 : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. (1989).
A two-band model for the calculation of clear-sky solar irradiance,
illuminance, and photosynthetically active radiation at the Earth's
surface.

Gueymard, C. (1993).
CPCR2 model with revised optical masses.
Solar Energy.

Chris' Fortran code.

### `csr(sza, datetime, pressure, broadbandaod, wv, ozone, albedo)`

CSR 2012 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `broadbandaod` (*np.ndarray*) — Broadband aerosol optical depth
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ozone` (*np.ndarray*) — Total ozone amount [atm-cm]
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghCSR, EbnCSR, EdhCSR]` (*list of np.ndarray*) — EghCSR : Global horizontal irradiance [W m-2] EbnCSR : Direct normal irradiance [W m-2] EdhCSR : Diffuse horizontal irradiance [W m-2]

**References**

Maxwell, E. L. (1998).
METSTAT—The solar radiation model used in the production of the
National Solar Radiation Data Base.

Gueymard, C. A. (2012).
Clear-sky irradiance predictions for solar resource mapping and
large-scale applications: Improved validation methodology and
detailed performance analysis of 18 broadband radiative models.

**Notes**

The only difference between CSR and METSTAT is the C_coeff term
used in the direct normal irradiance calculation.

### `dai(sza, datetime, pressure, wv, ang_beta, ang_alpha)`

Dai 2014 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ang_alpha` (*np.ndarray*) — Angstrom exponent

**Returns**

- `[EghDAI, EbnDAI, EdhDAI]` (*list of np.ndarray*) — EghDAI : Global horizontal irradiance [W m-2] EbnDAI : Direct normal irradiance [W m-2] EdhDAI : Diffuse horizontal irradiance [W m-2]

**References**

Dai, Q., & Fang, X. (2014).
A simple model to predict solar radiation under clear sky conditions.

### `dogniaux(sza, datetime, pressure, decli, TL2_series)`

Dogniaux 1976 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `decli` (*np.ndarray*) — Solar declination angle [degrees]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `[EghDOGNIAUX, EbnDOGNIAUX, EdhDOGNIAUX]` (*list*) — EghDOGNIAUX : list of global horizontal irradiance series EbnDOGNIAUX : list of direct normal irradiance series EdhDOGNIAUX : list of diffuse horizontal irradiance series The order of each list follows the input order of TL2_series.

**References**

Dogniaux, B. (1976).
Programme de calcul des eclairements solaires energetiques et lumineux
de surfaces orientees et inclinees.

Dogniaux, B. (1970).
Variations qualitatives et quantitatives des composantes du rayonnement
solaire sur une surface horizontale par ciel serein en fonction du
trouble atmospherique.

Dogniaux, B. (1975).
Variations geographiques et climatiques des expositions energetiques
solaires sur des surfaces receptrices horizontales et verticales.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

### `dpp(sza)`

Original DPP clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `[Eghdpp, Ebndpp, Edhdpp]` (*list of np.ndarray*) — Eghdpp : Global horizontal irradiance [W m-2] Ebndpp : Direct normal irradiance [W m-2] Edhdpp : Diffuse horizontal irradiance [W m-2]

**References**

Daneshyar, M. (1978).
Solar radiation statistics for Iran.
Sol. Energy;(United States), 21(4).

Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M.,
Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models
using routine hourly global irradiance measurements in Romania.
Renewable Energy, 55, 85-103.

### `elmghouchi(sza, datetime)`

Original El Mghouchi clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth automatically.

**Returns**

- `[EghElMghouchi, EbnElMghouchi, EdhElMghouchi]` (*list of np.ndarray*) — EghElMghouchi : Global horizontal irradiance [W m-2] EbnElMghouchi : Direct normal irradiance [W m-2] EdhElMghouchi : Diffuse horizontal irradiance [W m-2]

**References**

El Mghouchi, Y., El Bouardi, A., Choulli, Z., & Ajzoul, T. (2014).
New model to estimate and evaluate the solar radiation.
International Journal of Sustainable Built Environment, 3(2), 225-234.

### `esra(sza, datetime, altitude, TL2_series)`

Original ESRA clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `altitude` (*np.ndarray*) — Location altitude [m]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `[EghEsra, EbnEsra, EdhEsra]` (*list*) — EghEsra : list of global horizontal irradiance series EbnEsra : list of direct normal irradiance series EdhEsra : list of diffuse horizontal irradiance series The order of each list follows the input order of TL2_series.

**References**

Rigollier, C., Bauer, O., & Wald, L. (2000). On the clear sky model of the ESRA—European Solar Radiation Atlas—with respect to the Heliosat method.
Solar Energy, 68(1), 33-48.

### `furich(sza, datetime, altitude)`

Original Fu and Rich clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `altitude` (*np.ndarray*) — Location altitude [m]

**Returns**

- `[Eghfurich, Ebnfurich, Edhfurich]` (*list of np.ndarray*) — Eghfurich : Global horizontal irradiance [W m-2] Ebnfurich : Direct normal irradiance [W m-2] Edhfurich : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
Solar Energy, 86(8), 2145-2169.

### `heliosat(sza, datetime, press, TL2_series)`

Original Heliosat-1 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `[EghHeliosat1, EbnHeliosat1, EdhHeliosat1]` (*list*) — EghHeliosat1 : list of global horizontal irradiance series EbnHeliosat1 : list of direct normal irradiance series EdhHeliosat1 : list of diffuse horizontal irradiance series The order of each list follows the input order of TL2_series.

**References**

Hammer, A., Heinemann, D., Hoyer, C., Kuhlemann, R., Lorenz, E., Müller, R., & Beyer, H. G. (2003).
Solar energy assessment using remote sensing technologies.
Remote Sensing of Environment, 86(3), 423-432.

Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
Solar Energy, 86(8), 2145-2169.

### `heliosat2(sza, datetime, altitude, TL2_series)`

Original Heliosat-2 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `altitude` (*np.ndarray*) — Location altitude [m]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `[EghHeliosat2, EbnHeliosat2, EdhHeliosat2]` (*list*) — EghHeliosat2 : list of global horizontal irradiance series EbnHeliosat2 : list of direct normal irradiance series EdhHeliosat2 : list of diffuse horizontal irradiance series The order of each list follows the input order of TL2_series.

**References**

Lefevre, M., Albuisson, M., & Wald, L. (2002).
Joint report on interpolation scheme 'meteosat' and database 'climatology' i (meteosat).
SoDa Deliverable D3-8 and D5-1-4. Internal document.

### `hlj(sza, Dayth, Year, altitude)`

Original HLJ clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `Dayth` (*np.ndarray*) — Day number ranging from 1 to 365
- `Year` (*np.ndarray*) — Year
- `altitude` (*np.ndarray*) — Location altitude [m]

**Returns**

- `[Eghhlj, Ebnhlj, Edhhlj]` (*list of np.ndarray*) — Eghhlj : Global horizontal irradiance [W m-2] Ebnhlj : Direct normal irradiance [W m-2] Edhhlj : Diffuse horizontal irradiance [W m-2]

**References**

Hottel, H. C. (1976). A simple model for estimating the transmittance of direct solar radiation through clear atmospheres.
Solar Energy, 18(2), 129-134.

Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
Solar Energy, 86(8), 2145-2169.

### `hoyt(sza, datetime, press, albedo, ang_beta, ozone, wv)`

Original Hoyt clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghhoyt, Ebnhoyt, Edhhoyt]` (*list of np.ndarray*) — Eghhoyt : Global horizontal irradiance [W m-2] Ebnhoyt : Direct normal irradiance [W m-2] Edhhoyt : Diffuse horizontal irradiance [W m-2]

**References**

Iqbal, M. (2012). An introduction to solar radiation.
Elsevier.

### `hs(sza)`

HS 1976 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `[EghHS, EbnHS, EdhHS]` (*list of np.ndarray*) — EghHS : Global horizontal irradiance [W m-2] EbnHS : Direct normal irradiance [W m-2] EdhHS : Diffuse horizontal irradiance [W m-2]

**References**

Schulze, R. E. (1976).
A physically based method of estimating solar radiation from suncards.

Hourwitz, B. (1945).
Insolation in relation to cloudiness and cloud density.

Hourwitz, B. (1946).
Insolation in relation to cloud type.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

**Notes**

Direct normal irradiance is based on the Hourwitz model, while diffuse
horizontal irradiance is based on the Schulze formulation.

### `ideriah(sza, datetime, pressure, wv, ang_alpha, ang_beta)`

Ideriah 1981 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient

**Returns**

- `[EghIDERIAH, EbnIDERIAH, EdhIDERIAH]` (*list of np.ndarray*) — EghIDERIAH : Global horizontal irradiance [W m-2] EbnIDERIAH : Direct normal irradiance [W m-2] EdhIDERIAH : Diffuse horizontal irradiance [W m-2]

**References**

Ideriah, F. J. K. (1981).
A model for calculating direct and diffuse solar radiation.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

### `ineichen_2008(sza, datetime, press, aod700, wv)`

Original simplified Solis clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `aod700` (*np.ndarray*) — Aerosol optical depth at 700 nm [dimensionless]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghsolis, Ebnsolis, Edhsolis]` (*list of np.ndarray*) — Eghsolis : Global horizontal irradiance [W m-2] Ebnsolis : Direct normal irradiance [W m-2] Edhsolis : Diffuse horizontal irradiance [W m-2]

**References**

Ineichen, P. (2008). A broadband simplified version of the Solis clear sky model.
Solar Energy, 82(8), 758-762.

### `ineichen_2018(sza, datetime, pressure, wv, aod550)`

Advanced Solis 2018 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `aod550` (*np.ndarray*) — Aerosol optical depth at 550 nm

**Returns**

- `[EghINEICHEN2018, EbnINEICHEN2018, EdhINEICHEN2018]` (*list of np.ndarray*) — EghINEICHEN2018 : Global horizontal irradiance [W m-2] EbnINEICHEN2018 : Direct normal irradiance [W m-2] EdhINEICHEN2018 : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard's code, developed following discussions with Ineichen and
including corrections to the model implementation.

Ineichen, P. (2018).
High Turbidity Solis Clear Sky Model: Development and Validation.

### `ineichenperez(zsa, datetime, altitude, TL2_series)`

Original Ineichen & Perez clear-sky model.

**Parameters**

- `zsa` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `altitude` (*np.ndarray*) — Location altitude [m]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `Eghip` (*list[np.ndarray]*) — Global horizontal irradiance series computed from the same model using each TL2 input in TL2_series, in the same order.

**References**

Ineichen, P., & Perez, R. (2002).
A new airmass independent formulation for the Linke turbidity coefficient.
Solar Energy, 73(3), 151-157.

### `iqbal_a(sza, datetime, pressure, wv, ozone, ang_alpha, ang_beta, albedo)`

Iqbal-A 1983 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ozone` (*np.ndarray*) — Total ozone amount [atm-cm]
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghIQBALA, EbnIQBALA, EdhIQBALA]` (*list of np.ndarray*) — EghIQBALA : Global horizontal irradiance [W m-2] EbnIQBALA : Direct normal irradiance [W m-2] EdhIQBALA : Diffuse horizontal irradiance [W m-2]

**References**

Iqbal, M. (1983).
An Introduction to Solar Radiation.
Chapter 7.5, Model A.

### `iqbal_b(sza, datetime, pressure, wv, ozone, ang_beta, albedo)`

Iqbal-B 1983 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ozone` (*np.ndarray*) — Total ozone amount [atm-cm]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghIQBALB, EbnIQBALB, EdhIQBALB]` (*list of np.ndarray*) — EghIQBALB : Global horizontal irradiance [W m-2] EbnIQBALB : Direct normal irradiance [W m-2] EdhIQBALB : Diffuse horizontal irradiance [W m-2]

**References**

Iqbal, M. (1983).
An Introduction to Solar Radiation.
Chapter 7.5, Model B.

### `iqbal_c(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, wv)`

Original Iqbal-C clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghiqbalc, Ebniqbalc, Edhiqbalc]` (*list of np.ndarray*) — Eghiqbalc : Global horizontal irradiance [W m-2] Ebniqbalc : Direct normal irradiance [W m-2] Edhiqbalc : Diffuse horizontal irradiance [W m-2]

**References**

Iqbal, M. (2012). An introduction to solar radiation.
Elsevier.

### `janjai(sza, datetime, altitude, ang_alpha, ang_beta, ozone, wv)`

Original Janjai clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `altitude` (*np.ndarray*) — Location altitude [m]
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghjanjai, Ebnjanjai, Edhjanjai]` (*list of np.ndarray*) — Eghjanjai : Global horizontal irradiance [W m-2] Ebnjanjai : Direct normal irradiance [W m-2] Edhjanjai : Diffuse horizontal irradiance [W m-2]

**References**

Janjai, S., Sricharoen, K., & Pattarapanitchai, S. (2011). Semi-empirical models for the estimation of clear sky solar global and direct normal irradiances in the tropics.
Applied Energy, 88(12), 4749-4755.

### `josefsson(sza, datetime, press, wv, albedo)`

Original Josefsson clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]
- `albedo` (*np.ndarray*) — Surface/ground albedo

**Returns**

- `[EghJosefsson, EbnJosefsson, EdhJosefsson]` (*list of np.ndarray*) — EghJosefsson : Global horizontal irradiance [W m-2] EbnJosefsson : Direct normal irradiance [W m-2] EdhJosefsson : Diffuse horizontal irradiance [W m-2]

**References**

Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `kasm(sza, datetime, press, wv)`

Original KASM clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghKASM, EbnKASM, EdhKASM]` (*list of np.ndarray*) — EghKASM : Global horizontal irradiance [W m-2] EbnKASM : Direct normal irradiance [W m-2] EdhKASM : Diffuse horizontal irradiance [W m-2]

**References**

Kasten, F. (1984). Parametrisierung der Globalstahlung durch Bedeckungsgrad und Trubungsfaktor.
Annalen der Meteorologie Neue, 20, 49-50.

Badescu, V. (1997). Verification of some very simple clear and cloudy sky models to evaluate global solar irradiance.
Solar Energy, 61(4), 251-264.

Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements in Romania.
Renewable Energy, 55, 85-103.

### `kasten(sza, datetime, altitude, TL2_series)`

Original Kasten clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input used to calculate day-of-year and year.
- `altitude` (*np.ndarray*) — Location altitude [m]
- `TL2_series` (*list[np.ndarray]*) — A list of TL2 series. Input as many TL2 series as needed. Each TL2 represents Linke turbidity at air mass = 2.

**Returns**

- `Eghkasten` (*list[np.ndarray]*) — Global horizontal irradiance series computed from the same Kasten formula using each TL2 input in TL2_series, in the same order.

**References**

Ineichen, P., & Perez, R. (2002).
A new airmass independent formulation for the Linke turbidity coefficient.
Solar Energy, 73(3), 151-157.

### `king(sza, datetime, press, albedo, wv, ang_beta)`

Original King clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*np.ndarray*) — Datetime data. It is used to calculate Dayth and Year automatically.
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient

**Returns**

- `[EghKing, EbnKing, EdhKing]` (*list of np.ndarray*) — EghKing : Global horizontal irradiance [W m-2] EbnKing : Direct normal irradiance [W m-2] EdhKing : Diffuse horizontal irradiance [W m-2]

**References**

King, R., & Buckius, R. O. (1979). Direct solar transmittance for a clear sky.
Solar Energy, 22(3), 297-301.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `kumar(sza, datetime, press)`

Original Kumar clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]

**Returns**

- `[Eghkumar, Ebnkumar, Edhkumar]` (*list of np.ndarray*) — Eghkumar : Global horizontal irradiance [W m-2] Ebnkumar : Direct normal irradiance [W m-2] Edhkumar : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
Solar Energy, 86(8), 2145-2169.

### `mac1(sza, datetime, pressure, wv, albedo)`

MAC1 1982 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghMAC1, EbnMAC1, EdhMAC1]` (*list of np.ndarray*) — EghMAC1 : Global horizontal irradiance [W m-2] EbnMAC1 : Direct normal irradiance [W m-2] EdhMAC1 : Diffuse horizontal irradiance [W m-2]

**References**

Davies, J. A., & McKay, D. C. (1982).
Estimating solar irradiance and components.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

### `mac2(sza, datetime, press, albedo, ang_alpha, ang_beta, wv)`

Original MAC2 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghmac, Ebnmac, Edhmac]` (*list of np.ndarray*) — Eghmac : Global horizontal irradiance [W m-2] Ebnmac : Direct normal irradiance [W m-2] Edhmac : Diffuse horizontal irradiance [W m-2]

**References**

Davies, J. A., & McKay, D. C. (1982). Estimating solar irradiance and components.
Solar Energy, 29(1), 55-64.

### `metstat(sza, datetime, press, albedo, broadbandaod, ozone, wv)`

Original METSTAT clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `broadbandaod` (*np.ndarray*) — Broadband aerosol optical depth [dimensionless]
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghMetstat, EbnMetstat, EdhMetstat]` (*list of np.ndarray*) — EghMetstat : Global horizontal irradiance [W m-2] EbnMetstat : Direct normal irradiance [W m-2] EdhMetstat : Diffuse horizontal irradiance [W m-2]

**References**

Maxwell, E. L. (1998). METSTAT—The solar radiation model used in the production of the National Solar Radiation Data Base (NSRDB).
Solar Energy, 62(4), 263-279.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `mmac(sza, datetime, press, albedo, broadbandaod, wv)`

Original MMAC clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `broadbandaod` (*np.ndarray*) — Broadband aerosol optical depth [dimensionless]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghmmac, Ebnmmac, Edhmmac]` (*list of np.ndarray*) — Eghmmac : Global horizontal irradiance [W m-2] Ebnmmac : Direct normal irradiance [W m-2] Edhmmac : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. (1993). Critical analysis and performance assessment of clear sky solar irradiance models using theoretical and measured data.
Solar Energy, 51(2), 121-138.
Gueymard, C. A. (2003). Direct solar transmittance and irradiance predictions with broadband models. Part I: detailed theoretical performance assessment.
Solar Energy, 74(5), 355-379.
Davies, J. A., & McKay, D. C. (1982). Estimating solar irradiance and components.
Solar Energy, 29(1), 55-64.

### `modified_iqbalc(sza, datetime, altitude, albedo, ang_alpha, ang_beta, ozone, wv)`

Original modified Iqbal-C clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `Dayth` (*np.ndarray*) — Day number ranging from 1 to 365
- `Year` (*np.ndarray*) — Year
- `altitude` (*np.ndarray*) — Location altitude [m]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghMiqbalc, EbnMiqbalc, EdhMiqbalc]` (*list of np.ndarray*) — EghMiqbalc : Global horizontal irradiance [W m-2] EbnMiqbalc : Direct normal irradiance [W m-2] EdhMiqbalc : Diffuse horizontal irradiance [W m-2]

**References**

Bird, R. E., & Hulstrom, R. L. (1981). Simplified clear sky model for direct and diffuse insolation on horizontal surfaces
(No. SERI/TR-642-761). Solar Energy Research Inst., Golden, CO (USA).
Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
Solar Energy, 86(8), 2145-2169.

### `mrmv5(sza, datetime, press, albedo, ang_beta, ozone, wv)`

Original MRMv5 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghMRM5, EbnMRM5, EdhMRM5]` (*list of np.ndarray*) — EghMRM5 : Global horizontal irradiance [W m-2] EbnMRM5 : Direct normal irradiance [W m-2] EdhMRM5 : Diffuse horizontal irradiance [W m-2]

**References**

Kambezidis, H. D., & Psiloglou, B. E. (2008). The meteorological radiation model (MRM): advancements and applications.
In Modeling solar radiation at the earth’s surface (pp. 357-392). Springer, Berlin, Heidelberg.

### `mrmv6(sza, datetime, press, albedo, aod550, ozone, wv)`

Original MRMv6.1 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `aod550` (*np.ndarray*) — Aerosol optical depth at 550 nm [dimensionless]
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghMRM61, EbnMRM61, EdhMRM61]` (*list of np.ndarray*) — EghMRM61 : Global horizontal irradiance [W m-2] EbnMRM61 : Direct normal irradiance [W m-2] EdhMRM61 : Diffuse horizontal irradiance [W m-2]

**References**

Kambezidis, H. D., & Psiloglou, B. E. (2008). The meteorological radiation model (MRM): advancements and applications.
In Modeling solar radiation at the earth’s surface (pp. 357-392). Springer, Berlin, Heidelberg.
Kambezidis, H. D., Psiloglou, B. E., Karagiannis, D., Dumka, U. C., & Kaskaoutis, D. G. (2016).
Recent improvements of the Meteorological Radiation Model for solar irradiance estimates under all-sky conditions.
Renewable Energy, 93, 142-158.
Kambezidis, H. D., Psiloglou, B. E., Karagiannis, D., Dumka, U. C., & Kaskaoutis, D. G. (2017).
Meteorological Radiation Model (MRM v6.1): Improvements in diffuse radiation estimates and a new approach for implementation of cloud products.
Renewable and Sustainable Energy Reviews, 74, 616-637.
Author's Fortran code.

**Notes**

This model requires the external file 'AEROSOL-TABLE.txt' in the current working directory.

### `paltridge(sza)`

Paltridge 1976 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `[EghPALTRIDGE, EbnPALTRIDGE, EdhPALTRIDGE]` (*list of np.ndarray*) — EghPALTRIDGE : Global horizontal irradiance [W m-2] EbnPALTRIDGE : Direct normal irradiance [W m-2] EdhPALTRIDGE : Diffuse horizontal irradiance [W m-2]

**References**

Paltridge, G. W., & Platt, C. M. R. (1976).
Radiative Processes in Meteorology and Climatology.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

### `paulescu(sza, datetime, altitude, press, ang_beta, ozone, wv)`

Original Paulescu & Schlett clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `altitude` (*np.ndarray*) — Location altitude [m]
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghPaulescu, EbnPaulescu, EdhPaulescu]` (*list of np.ndarray*) — EghPaulescu : Global horizontal irradiance [W m-2] EbnPaulescu : Direct normal irradiance [W m-2] EdhPaulescu : Diffuse horizontal irradiance [W m-2]

**References**

Paulescu, M., & Schlett, Z. (2003). A simplified but accurate spectral solar irradiance model.
Theoretical and Applied Climatology, 75(3-4), 203-212.

### `perrin(sza, datetime, press, ang_beta, wv, ozone)`

Original Perrin clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]

**Returns**

- `[EghPerrin, EbnPerrin, EdhPerrin]` (*list of np.ndarray*) — EghPerrin : Global horizontal irradiance [W m-2] EbnPerrin : Direct normal irradiance [W m-2] EdhPerrin : Diffuse horizontal irradiance [W m-2]

**References**

de Brichambaut, C. P. (1975). Estimation des Ressources Energétiques en France.
Cahiers de l’AFEDES, (1).
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `pr(sza, datetime, press, albedo, ang_beta, ozone, wv)`

Original PR clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghPR, EbnPR, EdhPR]` (*list of np.ndarray*) — EghPR : Global horizontal irradiance [W m-2] EbnPR : Direct normal irradiance [W m-2] EdhPR : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. A. (2003). Direct solar transmittance and irradiance predictions with broadband models. Part I: detailed theoretical performance assessment.
Solar Energy, 74(5), 355-379.
Psiloglou, B. E., Santamouris, M., & Asimakopoulos, D. N. (2000). Atmospheric broadband model for computation of solar radiation at the Earth’s surface.
Application to Mediterranean climate. Pure and Applied Geophysics, 157(5), 829-860.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `psim(sza, datetime, press, albedo, ang_beta, wv)`

Original PSIM clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghPSIM, EbnPSIM, EdhPSIM]` (*list of np.ndarray*) — EghPSIM : Global horizontal irradiance [W m-2] EbnPSIM : Direct normal irradiance [W m-2] EdhPSIM : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. (1993). Mathermatically integrable parameterization of clear-sky beam and global irradiances and its use in daily irradiation applications.
Solar Energy, 50(5), 385-397.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `rest2v5(sza, datetime, press, albedo, ang_alpha, ang_beta, ozone, NO2, wv)`

Original REST2 version 5 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `Dayth` (*np.ndarray*) — Day number ranging from 1 to 365
- `Year` (*np.ndarray*) — Year
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `NO2` (*np.ndarray*) — Total columnar amount of nitrogen dioxide [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[Eghrest2, Ebnrest2, Edhrest2]` (*list of np.ndarray*) — Eghrest2 : Global horizontal irradiance [W m-2] Ebnrest2 : Direct normal irradiance [W m-2] Edhrest2 : Diffuse horizontal irradiance [W m-2]

**References**

Gueymard, C. A. (2008). REST2: High-performance solar radiation model for cloudless-sky irradiance, illuminance,
and photosynthetically active radiation–Validation with a benchmark dataset.
Solar Energy, 82(3), 272-285.

### `rsc(sza, datetime, press, albedo, ang_beta, wv)`

Original RSC clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `albedo` (*np.ndarray*) — Surface/ground albedo
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghRSC, EbnRSC, EdhRSC]` (*list of np.ndarray*) — EghRSC : Global horizontal irradiance [W m-2] EbnRSC : Direct normal irradiance [W m-2] EdhRSC : Diffuse horizontal irradiance [W m-2]

**References**

Carroll, J. J. (1985). Global transmissivity and diffuse fraction of solar radiation for clear and cloudy skies as measured and as predicted by bulk transmissivity models.
Solar Energy, 35(2), 105-118.
Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
in Romania. Renewable Energy, 55, 85-103.

### `schulze(sza)`

Original Schulze clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]

**Returns**

- `[Ebnschulze, Edhschulze, Eghschulze]` (*list of np.ndarray*) — Ebnschulze : Direct normal irradiance [W m-2] Edhschulze : Diffuse horizontal irradiance [W m-2] Eghschulze : Global horizontal irradiance [W m-2]

**References**

Schulze, R. E. (1976).
A physically based method of estimating solar radiation from suncards.
Agricultural Meteorology, 16(1), 85-101.

### `sharma(sza, datetime)`

Original Sharma clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input

**Returns**

- `[EghSharma, EbnSharma, EdhSharma]` (*list of np.ndarray*) — EghSharma : Global horizontal irradiance [W m-2] EbnSharma : Direct normal irradiance [W m-2] EdhSharma : Diffuse horizontal irradiance [W m-2]

**References**

Sharma, M. R., & Pal, R. S. (1965).
Interrelationships between total, direct, and diffuse solar radiation in the tropics.
Solar Energy, 9(4), 183-192.

Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M.,
Dumitrescu, A., ... & Rada, C. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models
using routine hourly global irradiance measurements in Romania.
Renewable Energy, 55, 85-103.

### `tj(sza, datetime)`

TJ 1957 clear-sky model.

**Inputs**

- `sza` (*array-like*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input

**Outputs**

Returns [EbnTJ, EdhTJ, EghTJ]:

- `EbnTJ` — Direct normal irradiance [W m-2]
- `EdhTJ` — Diffuse horizontal irradiance [W m-2]
- `EghTJ` — Global horizontal irradiance [W m-2]

**References**

Threlkeld, J. L., & Jordan, R. C. (1957).
Direct solar radiation available on clear days.
Heat., Piping Air Cond., 29(12).

Masters, G. M. (2013).
Renewable and efficient electric power systems.
John Wiley & Sons.

**Notes**

This function is translated directly from the original R code.

### `watt(sza, datetime, pressure, wv, ozone, ang_beta, ang_alpha, albedo)`

Watt 1978 clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input. It is used to calculate Dayth and Year automatically.
- `pressure` (*np.ndarray*) — Local barometric pressure [mb]
- `wv` (*np.ndarray*) — Total precipitable water vapour [cm]
- `ozone` (*np.ndarray*) — Total ozone amount [atm-cm]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ang_alpha` (*np.ndarray*) — Angstrom exponent
- `albedo` (*np.ndarray*) — Ground albedo

**Returns**

- `[EghWATT, EbnWATT, EdhWATT]` (*list of np.ndarray*) — EghWATT : Global horizontal irradiance [W m-2] EbnWATT : Direct normal irradiance [W m-2] EdhWATT : Diffuse horizontal irradiance [W m-2]

**References**

Watt, D. (1978).
On the Nature and Distribution of Solar Radiation.
Report for the U.S. Department of Energy.

Badescu, V. (2013).
Accuracy analysis for fifty-four clear-sky solar radiation models using
routine hourly global irradiance measurements in Romania.

### `yang(sza, datetime, press, ang_beta, ozone, wv)`

Original Yang clear-sky model.

**Parameters**

- `sza` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input
- `press` (*np.ndarray*) — Local barometric pressure [mb]
- `ang_beta` (*np.ndarray*) — Angstrom turbidity coefficient
- `ozone` (*np.ndarray*) — Total columnar amount of ozone [atm.cm]
- `wv` (*np.ndarray*) — Total columnar amount of water vapour [atm.cm]

**Returns**

- `[EghYang, EbnYang, EdhYang]` (*list of np.ndarray*) — EghYang : Global horizontal irradiance [W m-2] EbnYang : Direct normal irradiance [W m-2] EdhYang : Diffuse horizontal irradiance [W m-2]

**References**

Yang, K., & Koike, T. (2005). A general model to estimate hourly and daily solar radiation for hydrological studies.
Water Resources Research, 41(10).

### `yangwalsh(zsa, datetime)`

Original Yang & Walsh clear-sky model.

**Parameters**

- `zsa` (*np.ndarray*) — Solar zenith angle [radians]
- `datetime` (*array-like*) — Datetime input

**Returns**

- `Eghyangwalsh` (*np.ndarray*) — Global horizontal irradiance [W m-2]

**References**

Yang, D., Walsh, W. M., & Jirutitijaroen, P. (2014).
Estimation and applications of clear sky global horizontal irradiance at the equator.
Journal of Solar Energy Engineering, 136(3), 034505.

