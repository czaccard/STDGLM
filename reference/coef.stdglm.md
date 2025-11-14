# Coefficients in `stdglm` objects

Extracts coefficients from an `stdglm` object.

## Usage

``` r
# S3 method for class 'stdglm'
coef(object, type = "overall", ...)
```

## Arguments

- object:

  An `stdglm` object.

- type:

  A character string indicating the type of coefficients. Options are:

  - `overall`: Overall effects of varying coefficients.

  - `tvc`: Temporal effects of varying coefficients.

  - `svc`: Spatial effects of varying coefficients.

  - `stvc`: Spatio-temporal effects of varying coefficients.

  - `gamma`: Effects of covariates specified in the input `Z` (see
    [`stdglm`](https://czaccard.github.io/STDGLM/reference/stdglm.md)).

- ...:

  Additional arguments (currently ignored).

## Value

A dataframe with posterior mean and 95% credible intervals.

## Details

Extracts posterior mean and 95% credible intervals of the coefficients
according to their type.

## See also

[`stdglm`](https://czaccard.github.io/STDGLM/reference/stdglm.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `mod` is a fitted stdglm object
print(coef(mod, 'overall'))
print(coef(mod, 'gamma'))
betaST_post = coef(mod, 'stvc') # returns a data.frame
head(betaST_post)
} # }
```
