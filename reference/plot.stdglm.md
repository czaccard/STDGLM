# Plotting functions for `stdglm` objects

Plotting functions for `stdglm` objects.

## Usage

``` r
# S3 method for class 'stdglm'
plot(x, type = "fitted", ...)
```

## Arguments

- x:

  An `stdglm` object.

- type:

  A character string indicating the type of plot to generate. Options
  are:

  - `tvc`: Temporal effects of varying coefficients.

  - `svc`: Spatial effects of varying coefficients.

  - `stvc`: Spatio-temporal effects of varying coefficients.

  - `fitted`: Fitted values of the model.

- ...:

  Additional arguments passed to the plotting functions.

## Value

A ggplot object or a list of ggplot objects, depending on the type of
plot.

## Details

The `plot.stdglm` function dispatches to the appropriate plotting
function based on the `type` argument.

## See also

[`stdglm`](https://czaccard.github.io/STDGLM/reference/stdglm.md),
[`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `mod` is a fitted stdglm object
plot(mod, type = 'tvc')  # Plot temporal effects
plot(mod, type = 'svc', Coo_sf_obs, region)  # Plot spatial effects, 
# where `Coo_sf_obs` is a spatial object with observed coordinates and 
# `region` is a spatial region object, both from the `sf` package.
plot(mod, type = 'stvc', 1:4)  # Plot spatio-temporal effects
plot(mod, type = 'fitted', Y, 1)  # Plot fitted values
} # }
```
