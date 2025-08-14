#' @title Plotting functions for `stdglm` objects
#' @description Plotting functions for `stdglm` objects.
#' @param x A `stdglm` object.
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
#' @seealso \code{\link{stdglm}} for fitting the model.
#' 
#' @keywords plot
#' 
#' @importFrom rlang .data
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
#' @description Extracts the fitted values from a `stdglm` object.
#' @param object A `stdglm` object.
#' @param ... Additional arguments (not used).
#' @return A matrix of fitted values.
#' @details Extracts the posterior mean of the predictive distribution for the observed data points.
#' @export
#' 
fitted.stdglm <- function(object, ...) {
  return(object$ave$Yfitted_mean)
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

plot_svc = function(mod, Coo_sf_obs, region, Coo_sf_pred=NULL) {
  ave <- mod$ave
  ncov = dim(ave$Bspace_postmean)[3]
  q = stats::qnorm(.975)
  
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
    
    gg1 = ggplot2::ggplot() +
      ggplot2::geom_sf(data = region, fill='white') +
      ggplot2::geom_sf(ggplot2::aes(col=Bmean, shape=pred), Coo_sf, size=4) +
      ggplot2::labs(title = paste('Spatial Effect of beta', k-1), color = 'Mean')
    gg2 = ggplot2::ggplot() +
      ggplot2::geom_sf(data = region, fill='white') +
      ggplot2::geom_sf(ggplot2::aes(col=Bsd, shape=pred), Coo_sf, size=4) +
      ggplot2::labs(title = paste('Spatial Effect of beta', k-1), color = 'Std. Dev')
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
  
  gg = ggplot2::ggplot(df, ggplot2::aes(x=.data$Time)) +
    ggplot2::geom_line(ggplot2::aes(y=.data$Value, col=.data$Outcome, linetype=.data$Outcome)) +
    ggplot2::ggtitle('Observed vs. Fitted')
  return(gg)
}


