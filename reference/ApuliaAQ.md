# Apulia Air Quality Data

Daily measurements of some air pollutant concentrations, recorded by 51
ground-based monitoring stations in Apulia (Italy) in 2022, along with
related climate variables. This is a subset of data from the GRINS
AQCLIM dataset (see link below).

## Usage

``` r
ApuliaAQ
```

## Format

### `ApuliaAQ`

A data frame with 18,615 rows and 19 columns:

- AirQualityStation:

  Unique station identifier

- time:

  Date of measurement

- Longitude, Latitude:

  Geographic coordinates using the WGS-84 reference system

- Altitude:

  Altitude of the measurement station (m)

- AirQualityStationType:

  predominant type of emission sources in the station's vicinity

- AirQualityStationArea:

  type of area surrounding the station

- AQ_mean_NO2, AQ_mean_PM10, AQ_mean_PM2.5:

  Air pollutant concentrations (micrograms per cubic meter)

- CL_blh:

  Daily mean of the height of the atmosphere boundary layer (m)

- CL_lai_hv:

  Daily fixed value of high vegetation leaf area index (m2/m2)

- CL_lai_lv:

  Daily fixed value of low vegetation leaf area index (m2/m2)

- CL_rh:

  Daily mean of relative humidity (percent)

- CL_ssr:

  Daily maximum of surface solar radiation (J/m2)

- CL_t2m:

  Daily mean of air temperature at 2 meters (Celsius)

- CL_tp:

  Daily cumulative total precipitation (m)

- CL_winddir:

  Daily mode of wind direction (1=N, 2=E, 3=S, 4=W)

- CL_windspeed:

  Daily mean of wind speed (m/s)

## Source

<https://doi.org/10.5281/zenodo.15699805>
