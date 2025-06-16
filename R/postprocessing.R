#' @export
#'
plot.tvc = function(mod) {
  require(ggplot2)
  ave <- mod$ave
  ncov = dim(ave$Btime_postmean)[3]
  q = qnorm(.975)
  
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
    
    gg[[k]] = ggplot(df, aes(x=Time, col=pred)) +
      geom_line(aes(y= Bmean)) +
      geom_line(aes(y= B025), linetype='dashed') +
      geom_line(aes(y= B975), linetype='dashed') +
      labs(y=paste('beta', k-1), title = 'Temporal Effect')
    
  }
  return(gg)
}

#' @export
#'
plot.svc = function(mod, Coo_sf_obs, region, Coo_sf_pred=NULL) {
  require(ggplot2)
  require(ggpubr)
  ave <- mod$ave
  ncov = dim(ave$Bspace_postmean)[3]
  q = qnorm(.975)
  
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
    
    gg1 = ggplot() +
      geom_sf(data = region, fill='white') +
      geom_sf(aes(col=Bmean, shape=pred), Coo_sf, size=4) +
      labs(title = paste('Spatial Effect of beta', k-1), color = 'Mean')
    gg2 = ggplot() +
      geom_sf(data = region, fill='white') +
      geom_sf(aes(col=Bsd, shape=pred), Coo_sf, size=4) +
      labs(title = paste('Spatial Effect of beta', k-1), color = 'Std. Dev')
    gg[[k]] = ggarrange(gg1, gg2, nrow = 1, ncol = 2)
  }
  return(gg)
}

#' @export
#'
plot.stvc = function(mod, ids, pred = FALSE) {
  require(ggplot2)
  ave <- mod$ave
  ncov = dim(ave$Bspacetime_postmean)[3]
  q = qnorm(.975)
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
    df = df %>% filter(Space %in% ids)
    
    gg[[k]] = ggplot(df, aes(x=Time)) +
      geom_line(aes(y= Bmean)) +
      geom_line(aes(y= B025), linetype='dashed') +
      geom_line(aes(y= B975), linetype='dashed') +
      labs(title=paste('Spatio-Temporal Effect of beta', k-1)) +
      facet_wrap(~Space, nn, nn, labeller = 'label_both')
    
  }
  return(gg)
}

#' @export
#'
plot.fitted = function(mod, Y, id) {
  require(ggplot2)
  ave <- mod$ave
  
  df = data.frame(Time = 1:NCOL(Y),
                  Y = Y[id,],
                  Fitted = ave$Ypred_mean[id,])
  df = pivot_longer(df, Y:Fitted, names_to = "Outcome", values_to = 'Value')
  
  gg = ggplot(df, aes(x=Time)) +
    geom_line(aes(y=Value, col=Outcome)) +
    ggtitle('Observed vs. Fitted')
  return(gg)
}


