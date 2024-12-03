
# Helper functions for SiftingStats.Rmd

#### Functions altered from the precrec packge for improved visualization ####
# of ROC/PRC plots for classifier evaluation: 
get_pn_info <- function(object) {
  nps <- attr(object, "data_info")[["np"]]
  nns <- attr(object, "data_info")[["nn"]]
  
  is_consistant <- TRUE
  prev_np <- NA
  prev_nn <- NA
  np_tot <- 0
  nn_tot <- 0
  n <- 0
  for (i in seq_along(nps)) {
    np <- nps[i]
    nn <- nns[i]
    
    if ((!is.na(prev_np) && np != prev_np)
        ||  (!is.na(prev_nn) && nn != prev_nn)) {
      is_consistant <- FALSE
    }
    
    np_tot <- np_tot + np
    nn_tot <- nn_tot + nn
    prev_np <- np
    prev_nn <- nn
    n <- n + 1
  }
  
  avg_np <- np_tot / n
  avg_nn <- nn_tot / n
  
  prc_base <- avg_np / (avg_np + avg_nn)
  
  list(avg_np = avg_np, avg_nn = avg_nn, is_consistant = is_consistant,
       prc_base = prc_base)
  
}


geom_basic <- function(p, main, xlab, ylab, show_legend) {
  p <- p + ggplot2::theme_classic()
  p <- p + ggplot2::ggtitle(main)
  p <- p + ggplot2::xlab(xlab)
  p <- p + ggplot2::ylab(ylab)
  p <- p + ggplot2::theme(
    legend.title = element_text(size=12), #12 for square fig, 22 for wide
    text = element_text(size=16), #16 for squae, 24 for wide
    plot.title = element_text(size=14,face='bold', hjust = 0.5), )#14 for square 22 for wide
  if (!show_legend) {
    p <- p + ggplot2::theme(legend.position = "none")
  }
  
  p
}

make_rocprc_title <- function(object, pt) {
  pn_info <- get_pn_info(object)
  np <- pn_info$avg_np
  nn <- pn_info$avg_nn
  paste0(pt, " - P: ", np, ", N: ", nn)
}

geom_basic_roc <- function(p, object, show_legend = TRUE, add_np_nn = TRUE,
                           xlim, ylim, ratio, ...) {
  
  pn_info <- get_pn_info(object)
  
  if (add_np_nn && pn_info$is_consistant) {
    main <- make_rocprc_title(object, "ROC")
  } else {
    main <- "ROC"
  }
  
  p <- p + ggplot2::geom_abline(intercept = 0, slope = 1,colour = "red",
                                linetype = 'dotted',alpha=0.65, line)
  p <- set_coords(p, xlim, ylim, ratio)
  p <- geom_basic(p, main, "1 - Specificity", "Sensitivity", show_legend)
  
  p
}

geom_basic_prc <- function(p, object, show_legend = TRUE, add_np_nn = TRUE,
                           xlim, ylim, ratio, ...) {
  
  pn_info <- get_pn_info(object)
  
  if (add_np_nn && pn_info$is_consistant) {
    main <- make_rocprc_title(object, "Precision-Recall")
  } else {
    main <- "Precision-Recall"
  }
  
  p <- p + ggplot2::geom_hline(yintercept = pn_info$prc_base, colour = "red",
                               linetype = 'dotted',alpha=0.65,)
  p <- set_coords(p, xlim, ylim, ratio)
  p <- geom_basic(p, main, "Recall", "Precision", show_legend)
  
  p
}

set_coords <- function(p, xlim, ylim, ratio) {
  
  if (is.null(ratio))  {
    p <- p + ggplot2::coord_cartesian(xlim = xlim, ylim = ylim)
  } else {
    p <- p + ggplot2::coord_fixed(ratio = ratio, xlim = xlim, ylim = ylim)
  }
  
  p
}
get_plot = function(curves, ctype)
{
  plot_df = fortify(curves)
  ncats = length(unique(plot_df$modname))
  if (ncats==3){
    colors = brewer.pal(ncats,'Spectral')
    colors[2] = '#EBAF1A'#'#FCCC44'
    colors[3] = 'forestgreen'
    #colors = scico(6, palette = 'bam')
    colors = c(colors[2:3],colors[6])
  }else if (ncats==2){
    colors = brewer.pal(3,'Spectral')
    colors[2] = 'forestgreen'
    #colors = scico(6, palette = 'bam')
    #colors = c(colors[2],colors[6])
    #colors = c(colors[1],colors[4])
  }
  aucs = auc(curves)
  summ_aucs = aucs%>%
    group_by(modnames,curvetypes)%>%
    summarise(mean(aucs))
  plot_df = subset(plot_df, curvetype == ctype)
  xlim <- attr(curves[["rocs"]], "xlim")
  ylim <- attr(curves[["rocs"]], "ylim")
  p <- ggplot(plot_df, aes_string(x = 'x', y = 'y',color='modname'),alpha=0.5) + 
    geom_line(na.rm = TRUE,linewidth=1.5) + #plot line
    scale_color_manual(values=colors) +
    geom_ribbon(aes(ymin=ymin,ymax=ymax,fill=modname,color=modname),alpha=0.3,linetype=0) + 
    scale_fill_manual(values=colors) +guides(fill = "none")# add error bars
  xs = seq(0,1,length.out=ncats+2)[2:(ncats+1)]
  for (i in seq(1,ncats)){
    mod = unique(plot_df$modname)[i]
    p <- p + annotate('text',x=xs[i], y=0,
                      color=colors[i],
                      size=8,
                      label=round(summ_aucs[summ_aucs$modnames==mod&summ_aucs$curvetypes==ctype,
                                            'mean(aucs)'],2))
  }
  if (ctype=='ROC'){
    g0 = geom_basic_roc(p, curves, show_legend = T, add_np_nn = F,
                        curves_df=plot_df,xlim = xlim, ylim = ylim, ratio = 1)
  } else{
    g0 = geom_basic_prc(p, curves, show_legend = T, add_np_nn = F,
                        curves_df=plot_df,xlim = xlim, ylim = ylim, ratio = 1)
  }
  
  return(g0)
}

get_rocprc = function(curves)
{
  abl_roc = get_plot(curves, 'ROC')
  abl_prc = get_plot(curves, 'PRC')
  g = abl_roc+abl_prc+plot_layout(ncol=2,guides = 'collect') +
    plot_annotation(tag_levels = 'a')
  return(g)
}

#### functions to aggregate the data ####
get_all_data = function(chosen_data, strategy){
  exp_names = sort(unique(chosen_data$exp_name))
  seeds = sort(unique(chosen_data$seed))
  num_opts = length(exp_names)/length(seeds)
  scores1 = list()
  labels1 = list()
  dsids = vector()
  for (i in 1:length(seeds)) {
    scores1[[i]]=list()
    labels1[[i]]=list()
    dsids = c(dsids,rep(i,num_opts))
  }
  categories = vector()
  prev_category = strsplit(exp_names[1],'_')[[1]][1]
  counter=1
  for (exp_name in exp_names){
    category = strsplit(exp_name,'_')[[1]][1]
    tdf = chosen_data[chosen_data$exp_name==exp_name,]
    #print(nrow(tdf))
    seed_idx = which(seeds==unique(tdf$seed))
    counter = ifelse(category==prev_category,counter,counter+1)
    categories[counter] = category
    prev_category = category
    if (strategy=='classifier'){
      scores1[[seed_idx]][[counter]] = tdf$mean_score1
      
    } else if(strategy=='mixed'){
      arch = unique(chosen_data[chosen_data$exp_name==exp_name,'architecture'])
      if (arch=='classifier')       {
        scores1[[seed_idx]][[counter]] = tdf$mean_score1
      }else if (arch=='supervised anomaly'){
        scores1[[seed_idx]][[counter]] = -tdf$mean_scores
      }else{
        scores1[[seed_idx]][[counter]] = -tdf$mean_scores
        
      }
    }
    else {scores1[[seed_idx]][[counter]] = -tdf$mean_scores}
    labels1[[seed_idx]][[counter]] = tdf$label=='abnormal'
  }

  curve_data=mmdata(scores=scores1, labels=labels1, 
                    modname=rep(categories,length(seeds)),
                    dsids=dsids)
  
  return(curve_data)
}

get_data_rarity = function(df,sup=F)
{
  subsamples = sort(unique(df$subsample_size))
  sub_aucs = matrix(nrow=length(subsamples), ncol=3)
  seeds = unique(df$seed)
  scores = list()
  labels = list()
  dsids = vector()
  for (i in 1:length(seeds))
  {
    scores[[i]] = list()
    labels[[i]] = list()
    dsids = c(dsids,c(rep(seeds[i],length(subsamples))))
  }
  for(i in seq(1,length(subsamples))){
    
    sub_size = subsamples[i]
    sub_df = df[df$subsample_size==sub_size,]
    for (j in 1:length(seeds)){
      seed = seeds[j]
      
      sub_seed = sub_df[sub_df$seed==seed, ]
      if(sup){
        scores[[j]][[i]] = -sub_seed$mean_scores
        
      }else{
        scores[[j]][[i]] =sub_seed$mean_score1
      }
      
      labels[[j]][[i]] = sub_seed$label == 'abnormal'
    }
  }
  
  dat = mmdata(scores=scores, labels=labels, dsids=dsids,
               modname=as.character(
                 rep(subsamples,length(seeds))))
  return(dat)
}


get_aucs = function(data,plot=F,sup=F){
  dat = get_data_rarity(data,sup=sup)
  curves = evalmod(dat)
  aucs = auc(curves)
  if(plot){
    return(get_rocprc(curves))
  }else
  {
    return(aucs)
  }
}


get_method_dat = function(ours,traditional)
{
  our_auc_df = get_aucs(ours)
  our_auc_df['method'] = 'ours'
  trad_auc_df = get_aucs(traditional)
  trad_auc_df['method'] = 'random'
  auc_df = rbind(our_auc_df,trad_auc_df)
  return(auc_df)
}


get_raref_dat = function(sup.df,cls.df)
{
  our_auc_df = get_aucs(sup.df,sup=T)
  our_auc_df['method'] = 'supervised.NF'
  trad_auc_df = get_aucs(cls.df)
  trad_auc_df['method'] = 'classifier'
  auc_df = rbind(our_auc_df,trad_auc_df)
  return(auc_df)
}

get_all_auc_data = function(df){
  rarities = unique(df$rarity_percent)
  # at each rarity, we'll calculate the following:
  all_aucs = data.frame() # mean auprcs across the different seeds (for each subsample_size and method)
  all_aucs2 = data.frame() # mean auprc difference between the two methods across seeds (for each subsample_size)
  all_aucs3 = data.frame() # raw auprc differences between the two methods (for each random_seed and subsample_size)
  all_aucs0 = data.frame() # raw auprcs for each experiment (for each, random_seed, subsample_size, and method)
  for (rarity in rarities){
    print(rarity)
    rar_df = subset(df, df$rarity_percent==rarity)
    traditional_rar = subset(rar_df, rar_df$sampling_strategy=='traditional')
    ours_rar =  subset(rar_df,rar_df$sampling_strategy=='ours_v1')
    auc_df = get_method_dat(ours_rar,traditional_rar)
    colnames(auc_df) = c('subsample_size','seed','curvetypes','aucs','method')
    auc_agg = auc_df %>%
      group_by(method,subsample_size,curvetypes)%>%
      summarise(mean=mean(aucs),sd=sd(aucs),median=median(aucs),.groups='drop')%>%
      mutate(se_low=mean-sd,se_high=mean+sd)
    auc_df2 = dcast(auc_df,curvetypes+seed+subsample_size~method,value.var = 'aucs')
    auc_agg2 = auc_df2 %>%
      group_by(subsample_size,curvetypes)%>%
      summarise(mean_diff = mean(ours-random),.groups='drop')
    auc_agg3 = auc_df2 %>%
      group_by(subsample_size,seed,curvetypes)%>%
      summarise(diff = ours-random,.groups='drop')
    auc_agg$subsample_size = as.numeric(auc_agg$subsample_size)
    auc_agg$rarity = as.numeric(rarity)
    auc_agg2$rarity = as.numeric(rarity)
    auc_agg3$rarity = as.numeric(rarity)
    auc_df$rarity = as.numeric(rarity)
    all_aucs = rbind(all_aucs,auc_agg)
    all_aucs2 = rbind(all_aucs2,auc_agg2)
    all_aucs3 = rbind(all_aucs3,auc_agg3)
    all_aucs0 = rbind(all_aucs0, auc_df)
  }
  return(list(all_aucs,all_aucs2,all_aucs3,all_aucs0))
}

#### More visualization functions ####

prep_data = function(sub_aucs,use_median=F){
  mean_sub_aucs = sub_aucs
  mean_sub_prcs = subset(mean_sub_aucs,mean_sub_aucs$curvetypes=='PRC')
  mean_sub_prcs$curvetypes = NULL
  data=mean_sub_prcs
  data$method[data$method=='ours']='proposed'
  data$method=as.factor(data$method)
  if (use_median){
    colnames(data)[colnames(data)=='median']='median_auc'
  }else{
    colnames(data)[colnames(data)=='mean']='auc'
  }
  data$log_rarity=log10(data$rarity)
  return(data)
}

get_visreg = function(aucs,use_median=F,labeling_effort=200){
  data = prep_data(aucs,use_median)
  rarities = c(-3.931119,-3.6300887,-3.2041200,-2.6020600,-2.0000000,-1.3979400,-0.9208188)
  #sort(unique(data$log_rarity)))
  if (use_median){
    model=lm(median_auc~method*log_rarity*subsample_size,data=data)
    ylabel = 'median AuPRC'
  }else{#use mean like original analysis
    model=lm(auc~method*log_rarity*subsample_size,data=data)
    ylabel = 'AuPRC'
  }
  
  g = visreg(model,'log_rarity',by='method',
             cond=list(subsample_size=labeling_effort),overlay=T,gg=T)+
    scale_x_continuous(breaks=rarities,
                       limits=c(log10(1.171875e-04),log10(1.200000e-01)),
                       labels=function(x) parse(text=paste0(format(100*signif(10^x,2),
                                                                   drop0trailing=T,scientific=F),"*'%'")))+
    theme_classic()+
    scale_color_manual(values=c('indianred1','steelblue4'))+
    labs(x='behavior rarity',
         y=ylabel)+
    theme(strip.background = element_blank(),
          strip.text.x = element_blank(),
          text=element_text(size=20),
          panel.spacing = unit(1.5, 'cm'))
  g[["layers"]][[2]][["aes_params"]][["size"]]=3
  return(g)
}

get_sim_aucprc = function(scores, labels,cls=F, type="PRC"){
  if (!cls){
    scores = -scores
  }
  dat = mmdata(scores=scores,labels=labels)
  curves = evalmod(dat)
  aucs = auc(curves)
  wanted_aucs =aucs[aucs$curvetypes==type,]
  return(wanted_aucs$aucs)
}

# Get fancy colorful table:
get_color_table = function(mean_auprcs){
  # Display normalized mean AuPRCs for the synthetic datasets for different baseline rarities and behavior similarities.
  # Here we're doing a short slight of hand to get the color scheme to adhere to the numeric difference but display text of mean AuPRC&SD
  # disclaimer: there probably is a cleaner way to do this but this one worked for us.
  wide_prc_diff = mean_auprcs %>%
    dplyr::select(data_rarity,data_std,text_diff)%>% #text version
    tidyr::pivot_wider(names_from=data_std,
                values_from = text_diff)%>%
    mutate(data_rarity=paste0(100*data_rarity,'%'))
  
  wide_prc_diff2 = mean_auprcs %>%
    dplyr::select(data_rarity,data_std,mean_diff)%>% # numeric version
    tidyr::pivot_wider(names_from=data_std,
                values_from = mean_diff)%>%
    mutate(data_rarity=paste0(100*data_rarity,'%'))
  
  # Initiate text table:
  gt_prcs = gt(wide_prc_diff,rowname_col = 'data_rarity')%>% 
    data_color(palette='viridis')
  # Play around with the table layout (gt has nice documentation if you want to learn more):
  gt_prcs =  
    gt_prcs |>
    tab_stubhead(label = "data rarity")|>
    tab_spanner(label='data standard deviation (bigger more overlap)', columns = c('0.5','1.5','2.5','5'))
  
  # Initiate numeric table:
  gt_prcs2 = gt(wide_prc_diff2,rowname_col = 'data_rarity')%>%
    data_color(domain = c(0, 1),palette='viridis')
  # Match the layout to that of the text version:
  gt_prcs2 =  
    gt_prcs2  |>
    tab_stubhead(label = "data rarity")|>
    tab_spanner(label='data standard deviation (bigger more overlap)', columns = c('0.5','1.5','2.5','5'))
  # Now grab  the colors from the numeric version:
  style_list = gt_prcs2[["_styles"]][["styles"]]
  # And assign it to the text version:
  for (i in 1:length(style_list)){
    gt_prcs[["_styles"]][["styles"]][[i]] = gt_prcs2[["_styles"]][["styles"]][[i]]
  }
  return(gt_prcs)
}
