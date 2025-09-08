#' @title Plotting functions for `stdglm` objects
#' @description Plotting functions for `stdglm` objects.
#' @param x An `stdglm` object.
#' @param type A character string indicating the type of plot to generate. Options are:
#' \itemize{
#'   \item `tvc`: Temporal effects of varying coefficients.
#'   \item `svc`: Spatial effects of varying coefficients.
#'   \item `stvc`: Spatio-temporal effects of varying coefficients.
#'   \item `fitted`: Fitted values of the model.
#' }
#' @param ... Additional arguments passed to the plotting functions.
#' @return A ggplot object or a list of ggplot objects, depending on the type of plot.
#' @details The `plot.stdglm` function dispatches to the appropriate plotting function based on the `type` argument.
#' @examples
#' \dontrun{
#' # Assuming `mod` is a fitted stdglm object
#' plot(mod, type = 'tvc')  # Plot temporal effects
#' plot(mod, type = 'svc', Coo_sf_obs, region)  # Plot spatial effects, 
#' # where `Coo_sf_obs` is a spatial object with observed coordinates and 
#' # `region` is a spatial region object, both from the `sf` package.
#' plot(mod, type = 'stvc', 1:4)  # Plot spatio-temporal effects
#' plot(mod, type = 'fitted', Y, 1)  # Plot fitted values
#' }
#' @seealso \code{\link{stdglm}}, \code{\link[ggplot2]{ggplot}}
#' 
#' @keywords plot
#' @importFrom ggplot2 .data
#' 
#' @export
#'
plot.stdglm <- function(x, type = 'fitted', ...) {
  if (type == 'tvc') {
    return(plot_tvc(x))
  } else if (type == 'svc') {
    return(plot_svc(x, ...))
  } else if (type == 'stvc') {
    return(plot_stvc(x, ...))
  } else if (type == 'fitted') {
    return(plot_fitted(x, ...))
  } else {
    stop("Unknown plot type. Use 'tvc', 'svc', 'stvc', or 'fitted'.")
  }
}

#' @title Fitted values for `stdglm` objects
#' @description Extracts the fitted values from an `stdglm` object.
#' @param object An `stdglm` object.
#' @param ... Additional arguments (currently ignored).
#' @return A matrix of fitted values.
#' @details Extracts the posterior mean of the predictive distribution for the observed data points.
#' @seealso \code{\link{stdglm}}
#' @examples
#' \dontrun{
#' # Assuming `mod` is a fitted stdglm object
#' fitted_values <- fitted(mod)
#' summary(t(fitted_values[1:5,])) # Extract temporal summaries for the first 5 locations
#' }
#' @export
#' 
fitted.stdglm <- function(object, ...) {
  return(object$ave$Yfitted_mean)
}

#' @title Retrieve predictions for `stdglm` objects
#' @description Extracts out-of-sample predictions from an `stdglm` object.
#' @param object An `stdglm` object.
#' @param type A character string indicating the type of coefficients. Options are:
#' \itemize{
#'   \item `response_mat`: Posterior mean of the predictive distribution for the response variable. Returns a \emph{p_new}-by-\emph{t_new} matrix (default).
#'   \item `response_df`: Mean, standard deviation and 95 % credible interval for the response variable (default). Returns a dataframe.
#'   \item `tvc`: Mean, standard deviation and 95 % credible interval for the temporal effects of varying coefficients.
#'   \item `svc`: Mean, standard deviation and 95 % credible interval for the spatial effects of varying coefficients.
#'   \item `stvc`: Mean, standard deviation and 95 % credible interval for the spatio-temporal effects of varying coefficients.
#' }
#' @param Coo_sf_pred A simple feature object from package `sf` with the prediction points, whose geometry is used for spatial effects (optional).
#' @param ... Additional arguments (currently ignored).
#' @return Either a matrix or a dataframe or an `sf` object with posterior mean and 95% credible interval bounds. The function returns \code{NULL} if predictions are not available in the `stdglm` object.
#' @details Returns the posterior mean of the predictive distribution and the associated 95% credible intervals for the out-of-sample data points, and for the random variable specified in the input \code{type}.
#' @seealso \code{\link{stdglm}}, \code{\link[sf]{sf}}
#' @examples
#' \dontrun{
#' # Assuming `mod` is a fitted stdglm object
#' predictions <- predict(mod)
#' colMeans(predictions) # Get the average predictions across all locations
#' pred_df <- predict(mod, type = 'response_df')
#' head(pred_df)
#' }
#' @export
#' 
predict.stdglm <- function(object, type = 'response_mat', Coo_sf_pred=NULL, ...) {
  q = stats::qnorm(.975)
  if (is.null(object$ave$Ypred_mean)) {
    warning("No predictions available in the stdglm object. Please ensure the model was fitted with out-of-sample predictions.")
    return(NULL)
  }

  if (type == 'response_mat') {
    return(object$ave$Ypred_mean)
  } else if (type == 'response_df') {
    Ypred_mean = object$ave$Ypred_mean
    Ypred_sd = sqrt(object$ave$Ypred2_mean - Ypred_mean^2)
    out = expand.grid(Space=1:NROW(Ypred_mean), Time=1:NCOL(Ypred_mean))
    out$Mean=as.vector(Ypred_mean)
    out$sd=as.vector(Ypred_sd)
    out$ci_low = as.vector(Ypred_mean - q*Ypred_sd)
    out$ci_high = as.vector(Ypred_mean + q*Ypred_sd)
    if (!is.null(Coo_sf_pred)) {
      # Add spatial coordinates to the output
      out$geometry = rep(sf::st_geometry(Coo_sf_pred), NCOL(Ypred_mean))
      out = sf::st_as_sf(out)
    }
    return(out)
  } else if (type == 'tvc') {
    ncov = dim(object$ave$Btime_pred_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Btime_pred_postmean[1,,k]
      Bsd = sqrt(object$ave$Btime_pred2_postmean[1,,k] - object$ave$Btime_pred_postmean[1,,k]^2)
      df = data.frame(Coef= paste0('beta', k-1), steps_ahead=1:length(Bmean), 
                      Mean=Bmean, sd = Bsd,
                      ci_low = Bmean - q*Bsd, ci_high = Bmean + q*Bsd)
      out = rbind(out, df)
    }
    return(out)
  } else if (type == 'svc') {
    ncov = dim(object$ave$Bspace_pred_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Bspace_pred_postmean[,1,k]
      Bsd = sqrt(object$ave$Bspace_pred2_postmean[,1,k] - object$ave$Bspace_pred_postmean[,1,k]^2)
      df = data.frame(Coef= paste0('beta', k-1), Space=1:length(Bmean), 
                      Mean=Bmean, sd = Bsd,
                      ci_low = Bmean - q*Bsd, ci_high = Bmean + q*Bsd)
      out = rbind(out, df)
    }
    if (!is.null(Coo_sf_pred)) {
      # Add spatial coordinates to the output
      out$geometry = rep(sf::st_geometry(Coo_sf_pred), ncov)
      out = sf::st_as_sf(out)
    }
    return(out)
  } else if (type == 'stvc') {
    ncov = dim(object$ave$Bspacetime_pred_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Bspacetime_pred_postmean[,,k]
      Bsd = sqrt(object$ave$Bspacetime_pred2_postmean[,,k] - object$ave$Bspacetime_pred_postmean[,,k]^2)

      df = expand.grid(Coef= paste0('beta', k-1), Space=1:NROW(Bmean), Time=1:NCOL(Bmean))
      df$Mean=as.vector(Bmean)
      df$sd = as.vector(Bsd)
      df$ci_low = as.vector(Bmean - q*Bsd)
      df$ci_high = as.vector(Bmean + q*Bsd)
      out = rbind(out, df)
    }
    if (!is.null(Coo_sf_pred)) {
      # Add spatial coordinates to the output
      out$geometry = rep(sf::st_geometry(Coo_sf_pred), ncov*NCOL(Bmean))
      out = sf::st_as_sf(out)
    }
    return(out)
  } else {
    stop("Unknown prediction type. Use 'response_mat', 'response_df', 'tvc', 'svc', or 'stvc'.")
  }
}

#' @title Coefficients in `stdglm` objects
#' @description Extracts coefficients from an `stdglm` object.
#' @param object An `stdglm` object.
#' @param type A character string indicating the type of coefficients. Options are:
#' \itemize{
#'   \item `overall`: Overall effects of varying coefficients.
#'   \item `tvc`: Temporal effects of varying coefficients.
#'   \item `svc`: Spatial effects of varying coefficients.
#'   \item `stvc`: Spatio-temporal effects of varying coefficients.
#'   \item `gamma`: Effects of covariates specified in the input \code{Z} (see \code{\link{stdglm}}).
#' }
#' @param ... Additional arguments (currently ignored).
#' @return A dataframe with posterior mean and 95% credible intervals.
#' @details Extracts posterior mean and 95% credible intervals of the coefficients according to their type.
#' @seealso \code{\link{stdglm}}
#' @examples
#' \dontrun{
#' # Assuming `mod` is a fitted stdglm object
#' print(coef(mod, 'overall'))
#' print(coef(mod, 'gamma'))
#' betaST_post = coef(mod, 'stvc') # returns a data.frame
#' head(betaST_post)
#' }
#' @export
#' 
coef.stdglm <- function(object, type = 'overall', ...) {
  q = stats::qnorm(.975)
  if (type == 'overall') {
    b_overall = as.vector(object$ave$B_postmean)
    sdb_overall = sqrt(as.vector(object$ave$B2_postmean) - b_overall^2)
    out = data.frame(Mean=b_overall, sd = sdb_overall,
                     ci_low=b_overall - q*sdb_overall, 
                     ci_high=b_overall+q*sdb_overall)
    return(out)
  } else if (type == 'tvc') {
    ncov = dim(object$ave$Btime_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Btime_postmean[1,,k]
      Bsd = sqrt(object$ave$Btime2_postmean[1,,k] - object$ave$Btime_postmean[1,,k]^2)
      df = data.frame(Coef= paste0('beta', k-1), Time=1:length(Bmean), 
                      Mean=Bmean, sd = Bsd,
                      ci_low = Bmean - q*Bsd, ci_high = Bmean + q*Bsd)
      out = rbind(out, df)
    }
    return(out)
  } else if (type == 'svc') {
    ncov = dim(object$ave$Bspace_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Bspace_postmean[,1,k]
      Bsd = sqrt(object$ave$Bspace2_postmean[,1,k] - object$ave$Bspace_postmean[,1,k]^2)
      df = data.frame(Coef= paste0('beta', k-1), Space=1:length(Bmean), 
                      Mean=Bmean, sd = Bsd,
                      ci_low = Bmean - q*Bsd, ci_high = Bmean + q*Bsd)
      out = rbind(out, df)
    }
    return(out)
  } else if (type == 'stvc') {
    ncov = dim(object$ave$Bspacetime_postmean)[3]
    out = data.frame()
    for (k in 1:ncov) {
      Bmean = object$ave$Bspacetime_postmean[,,k]
      Bsd = sqrt(object$ave$Bspacetime2_postmean[,,k] - object$ave$Bspacetime_postmean[,,k]^2)

      df = expand.grid(Coef= paste0('beta', k-1), Space=1:NROW(Bmean), Time=1:NCOL(Bmean))
      df$Mean=as.vector(Bmean)
      df$sd = as.vector(Bsd)
      df$ci_low = as.vector(Bmean - q*Bsd)
      df$ci_high = as.vector(Bmean + q*Bsd)
      out = rbind(out, df)
    }
    return(out)
  } else if (type == 'gamma') {
    gamma = rowMeans(object$out$gamma)
    sdb_gamma = apply(object$out$gamma, 1, stats::sd)
    out = data.frame(Mean=gamma, sd = sdb_gamma,
                     ci_low=gamma - q*sdb_gamma, 
                     ci_high=gamma + q*sdb_gamma)
    return(out)
  } else {
    stop("Unknown coefficient type. Use 'overall', 'tvc', 'svc', 'stvc', or 'gamma'.")
  }
}




plot_tvc = function(mod) {
  ave <- mod$ave
  ncov = dim(ave$Btime_postmean)[3]
  q = stats::qnorm(.975)
  
  gg = list()
  for (k in 1:ncov) {
    Bmean = ave$Btime_postmean[1,,k]
    Bsd = sqrt(ave$Btime2_postmean[1,,k] - ave$Btime_postmean[1,,k]^2)
    df = data.frame(k=k, Time=1:length(Bmean), Bmean=Bmean,
                    B025 = Bmean - q*Bsd, B975 = Bmean + q*Bsd,
                    pred = 0)
    Bpredmean = ave$Btime_pred_postmean[1,,k]
    if (length(Bpredmean) != 0) {
      Bpredsd = sqrt(ave$Btime_pred2_postmean[1,,k] - ave$Btime_pred_postmean[1,,k]^2)
      dfpred = data.frame(k=k, Time=1:length(Bpredmean)+length(Bmean), Bmean=Bpredmean,
                      B025 = Bpredmean - q*Bpredsd, B975 = Bpredmean + q*Bpredsd,
                      pred = 1)
      df = rbind(df, dfpred)
    }
    df$pred = factor(df$pred, levels = c(0, 1), labels = c('Observed', 'Predicted'))
    df = droplevels(df)
    
    gg[[k]] = ggplot2::ggplot(df, ggplot2::aes(x=Time, col=pred)) +
      ggplot2::geom_line(ggplot2::aes(y= Bmean)) +
      ggplot2::geom_line(ggplot2::aes(y= B025), linetype='dashed') +
      ggplot2::geom_line(ggplot2::aes(y= B975), linetype='dashed') +
      ggplot2::labs(y=paste('beta', k-1), title = 'Temporal Effect')
    
  }
  return(gg)
}

plot_svc = function(mod, Coo_sf_obs, region=NULL, Coo_sf_pred=NULL) {
  ave <- mod$ave
  ncov = dim(ave$Bspace_postmean)[3]
  q = stats::qnorm(.975)
  geom_types = as.character(sf::st_geometry_type(Coo_sf_obs)[1])

  gg = list()
  for (k in 1:ncov) {
    Coo_sf = Coo_sf_obs
    Bmean = ave$Bspace_postmean[,1,k]
    Bsd = sqrt(ave$Bspace2_postmean[,1,k] - ave$Bspace_postmean[,1,k]^2)
    Coo_sf$Bmean = Bmean
    Coo_sf$Bsd = Bsd
    Coo_sf$B025 = Bmean - q*Bsd
    Coo_sf$B975 = Bmean + q*Bsd
    Coo_sf$pred = 0
    
    if (!is.null(Coo_sf_pred)) {
      Bpredmean = ave$Bspace_pred_postmean[,1,k]
      Coo_sf_pred$Bmean = Bpredmean
      Coo_sf_pred$Bsd = sqrt(ave$Bspace_pred2_postmean[,1,k] - ave$Bspace_pred_postmean[,1,k]^2)
      Coo_sf_pred$B025 = Bpredmean - q*Coo_sf_pred$Bsd
      Coo_sf_pred$B975 = Bpredmean + q*Coo_sf_pred$Bsd
      Coo_sf_pred$pred = 1
      Coo_sf = rbind(dplyr::select(Coo_sf, Bmean:pred), dplyr::select(Coo_sf_pred, Bmean:pred))
    }
    Coo_sf$pred = factor(Coo_sf$pred, levels = c(0, 1), labels = c('Observed', 'Predicted'))
    
    gg1 = ggplot2::ggplot()
    if (!is.null(region))
      gg1 = gg1 + ggplot2::geom_sf(data = region, fill='white')
    if (endsWith(geom_types, "POINT")) {
      gg1 = gg1 + ggplot2::geom_sf(ggplot2::aes(col=Bmean, shape=pred), Coo_sf, size=4)
    } else {
      gg1 = gg1 + ggplot2::geom_sf(ggplot2::aes(fill=Bmean), Coo_sf)
    }
    gg1 = gg1 + ggplot2::labs(title = paste('Spatial Effect of beta', k-1), color = 'Mean')
    gg2 = ggplot2::ggplot()
    if (!is.null(region))
      gg2 = gg2 + ggplot2::geom_sf(data = region, fill='white')
    if (endsWith(geom_types, "POINT")) {
      gg2 = gg2 + ggplot2::geom_sf(ggplot2::aes(col=Bsd, shape=pred), Coo_sf, size=4)
    } else {
      gg2 = gg2 + ggplot2::geom_sf(ggplot2::aes(fill=Bsd), Coo_sf)
    }
    gg2 = gg2 + ggplot2::labs(title = paste('Spatial Effect of beta', k-1), color = 'Std. Dev')
    gg[[k]] = ggpubr::ggarrange(gg1, gg2, nrow = 1, ncol = 2)
  }
  return(gg)
}


plot_stvc = function(mod, ids, pred = FALSE) {
  ave <- mod$ave
  ncov = dim(ave$Bspacetime_postmean)[3]
  q = stats::qnorm(.975)
  nn = ceiling(sqrt(length(ids)))
  
  gg = list()
  for (k in 1:ncov) {
    if (pred) {
      Bmean = ave$Bspacetime_pred_postmean[,,k]
      Bsd = sqrt(ave$Bspacetime_pred2_postmean[,,k] - ave$Bspacetime_pred_postmean[,,k]^2)
    } else {
      Bmean = ave$Bspacetime_postmean[,,k]
      Bsd = sqrt(ave$Bspacetime2_postmean[,,k] - ave$Bspacetime_postmean[,,k]^2)
    }
    
    df = expand.grid(k=k, Space=1:NROW(Bmean), Time=1:NCOL(Bmean))
    df$Bmean=as.vector(Bmean)
    df$B025 = as.vector(Bmean - q*Bsd)
    df$B975 = as.vector(Bmean + q*Bsd)
    df = dplyr::filter(df, Space %in% ids)
    
    gg[[k]] = ggplot2::ggplot(df, ggplot2::aes(x=Time)) +
      ggplot2::geom_line(ggplot2::aes(y= Bmean)) +
      ggplot2::geom_line(ggplot2::aes(y= B025), linetype='dashed') +
      ggplot2::geom_line(ggplot2::aes(y= B975), linetype='dashed') +
      ggplot2::labs(title=paste('Spatio-Temporal Effect of beta', k-1)) +
      ggplot2::facet_wrap(~Space, nn, nn, labeller = 'label_both')
    
  }
  return(gg)
}


plot_fitted = function(mod, Y, id) {
  ave <- mod$ave
  
  df = data.frame(Time = 1:NCOL(Y),
                  Y = Y[id,],
                  Fitted = ave$Yfitted_mean[id,])
  df = tidyr::pivot_longer(df, "Y":"Fitted", names_to = "Outcome", values_to = 'Value')
  
  gg = ggplot2::ggplot(df, ggplot2::aes(x=Time)) +
    ggplot2::geom_line(ggplot2::aes(y=.data$Value, col=.data$Outcome, linetype=.data$Outcome)) +
    ggplot2::ggtitle('Observed vs. Fitted')
  return(gg)
}


