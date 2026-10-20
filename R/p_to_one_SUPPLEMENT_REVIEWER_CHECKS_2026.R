# =============================================================================
# P-to-One ACE--Distance Correlation
# Reviewer-oriented supplementary analyses for KJAS manuscript v1
#
# DOES NOT rerun the completed P0--P12 main simulations.
#
# Track A: Replot publication figures from existing P6--P12 output.
# Track B: Oracle recovery for P6--P12 using the SAME stored seeds.
# Track C: P6/P8/P11/P12 robustness to predictor distribution/correlation.
# Track D: Additional residual vs wild-HC2 null-bootstrap calibration.
# Track E: Representative bootstrap power for P6/P8/P11/P12.
# Track F: Concrete bootstrap refinement (residual/Rademacher/Mammen HC2).
#
# Manuscript-only tasks (literature, formal estimand definitions, Algorithm 1)
# are intentionally NOT performed here and should be added to the TeX later.
# =============================================================================

required_packages <- c("acepack","future","future.apply","readxl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly=TRUE)
]
if (length(missing_packages)>0L) {
  install.packages(missing_packages, repos="https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(acepack); library(future); library(future.apply); library(readxl)
})

BASE_SEED <- 20260908L
ACE_SPAN <- 0.5
ACE_MAXIT <- 50
CF_FOLDS <- 2L

RUN_PROFILE <- "main"  # "main" or "quick"
if (RUN_PROFILE=="main") {
  ORACLE_REPS_PER_CELL <- 500L
  ROBUST_B <- 500L
  ROBUST_CALIBRATION_N <- 100000L
  NULL_BOOT_DATASETS <- 300L
  NULL_BOOT_B <- 299L
  POWER_BOOT_DATASETS <- 300L
  POWER_BOOT_B <- 299L
  CONCRETE_BOOT_B <- 999L
} else {
  ORACLE_REPS_PER_CELL <- 50L
  ROBUST_B <- 30L
  ROBUST_CALIBRATION_N <- 20000L
  NULL_BOOT_DATASETS <- 20L
  NULL_BOOT_B <- 49L
  POWER_BOOT_DATASETS <- 20L
  POWER_BOOT_B <- 49L
  CONCRETE_BOOT_B <- 99L
}

RUN_REPLOT_EXISTING <- TRUE
RUN_ORACLE_DIAGNOSTIC <- TRUE
RUN_ROBUSTNESS <- TRUE
RUN_NULL_BOOTSTRAP_CALIBRATION <- TRUE
RUN_EXTENDED_BOOTSTRAP_POWER <- TRUE
RUN_CONCRETE_BOOTSTRAP_REFINEMENT <- TRUE

NULL_BOOT_METHODS <- c("residual","wild_rademacher_hc2")
POWER_BOOT_METHODS <- c("residual")
CONCRETE_BOOT_METHODS <- c(
  "residual","wild_rademacher_hc2","wild_mammen_hc2"
)

REQUESTED_WORKERS <- 8L
available_cores <- tryCatch(future::availableCores(), error=function(e) 1L)
MAX_WORKERS <- max(1L,min(REQUESTED_WORKERS,max(1L,as.integer(available_cores)-1L)))
Sys.setenv(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
if (MAX_WORKERS>1L) {
  future::plan(future::multisession,workers=MAX_WORKERS)
} else {
  future::plan(future::sequential)
}
options(future.globals.maxSize=6*1024^3)
set.seed(BASE_SEED)

PROJECT_DIR <- Sys.getenv("PORTFOLIO_PROJECT_DIR", unset = getwd())
if (!dir.exists(PROJECT_DIR)) stop("PROJECT_DIR not found: ",PROJECT_DIR)
setwd(PROJECT_DIR)

EXT_ROOT <- file.path(PROJECT_DIR,"paper_extended_P6_P12_v1")
EXT_SIM_DIR <- file.path(EXT_ROOT,"simulation")
SUPP_ROOT <- file.path(PROJECT_DIR,"paper_supplement_reviewer_v1")
SUPP_CHECKPOINT <- file.path(SUPP_ROOT,"checkpoints")
SUPP_TABLES <- file.path(SUPP_ROOT,"tables")
SUPP_FIGURES <- file.path(SUPP_ROOT,"paper_outputs")
SUPP_CONCRETE <- file.path(SUPP_ROOT,"concrete")
for (d in c(SUPP_ROOT,SUPP_CHECKPOINT,SUPP_TABLES,SUPP_FIGURES,SUPP_CONCRETE)) {
  if (!dir.exists(d)) dir.create(d,recursive=TRUE)
}
LOG_FILE <- file.path(SUPP_ROOT,"run_log.txt")
log_message <- function(...) {
  line <- paste0(format(Sys.time(),"%Y-%m-%d %H:%M:%S")," | ",paste(...,collapse=""))
  cat(line,"\n"); cat(line,"\n",file=LOG_FILE,append=TRUE)
}

extended_condition_levels <- c(
  "P6_dense_quadratic_additive","P7_dense_heterogeneous_additive",
  "P8_chain_pairwise_interaction","P9_quadratic_single_index",
  "P10_threshold_additive","P11_mixed_main_interaction",
  "P12_highorder_sign_interaction"
)
extended_condition_ids <- c("P6","P7","P8","P9","P10","P11","P12")
names(extended_condition_ids) <- extended_condition_levels
scenario_colors <- c(
  P6_dense_quadratic_additive="#1B9E77",
  P7_dense_heterogeneous_additive="#D95F02",
  P8_chain_pairwise_interaction="#7570B3",
  P9_quadratic_single_index="#E7298A",
  P10_threshold_additive="#66A61E",
  P11_mixed_main_interaction="#E6AB02",
  P12_highorder_sign_interaction="#A6761D"
)
scenario_point_shapes <- setNames(c(16,17,15,18,0,1,2),extended_condition_levels)
scenario_line_types <- setNames(c(1,2,3,4,5,6,1),extended_condition_levels)

safe_mean <- function(x) { x<-x[is.finite(x)]; if(length(x)) mean(x) else NA_real_ }
safe_sd <- function(x) { x<-x[is.finite(x)]; if(length(x)>1L) sd(x) else NA_real_ }
safe_quantile <- function(x,p) {
  x<-x[is.finite(x)]
  if(length(x)) unname(quantile(x,p,names=FALSE,type=7)) else NA_real_
}
r2_to_noise_sd <- function(r2) sqrt((1-r2)/r2)

distance_correlation <- function(x,y) {
  x<-as.matrix(x); y<-as.matrix(y); n<-nrow(x)
  a<-as.matrix(dist(x)); b<-as.matrix(dist(y))
  A<-a-rowMeans(a)-matrix(colMeans(a),n,n,byrow=TRUE)+mean(a)
  B<-b-rowMeans(b)-matrix(colMeans(b),n,n,byrow=TRUE)+mean(b)
  xy<-max(mean(A*B),0); xx<-max(mean(A*A),0); yy<-max(mean(B*B),0)
  den<-sqrt(sqrt(xx*yy))
  if(!is.finite(den)||den<=.Machine$double.eps) return(0)
  sqrt(xy)/den
}

standardize_train_test <- function(x_train,x_test) {
  mu<-colMeans(x_train); sc<-apply(x_train,2,sd)
  sc[!is.finite(sc)|sc<=.Machine$double.eps]<-1
  list(
    train=sweep(sweep(x_train,2,mu,"-"),2,sc,"/"),
    test=sweep(sweep(x_test,2,mu,"-"),2,sc,"/")
  )
}

make_ace_control <- function(span=ACE_SPAN,maxit=ACE_MAXIT) {
  list(span=as.double(span),maxit=as.double(maxit))
}
fit_ace_safely <- function(x,y,span=ACE_SPAN) {
  tryCatch({
    fit<-acepack::ace(
      x=as.matrix(x),y=as.numeric(y),delrsq=1e-4,
      control=make_ace_control(span,ACE_MAXIT),
      on.error=function(message) stop(message)
    )
    tx<-as.matrix(fit$tx); ty<-as.numeric(fit$ty)
    list(success=TRUE,fit=fit,tx=tx,ty=ty,score=rowSums(tx))
  },error=function(e) list(success=FALSE,error_message=conditionMessage(e)))
}
make_interp_function <- function(x,z) {
  ok<-is.finite(x)&is.finite(z); x<-x[ok]; z<-z[ok]
  if(length(x)<4L||!is.finite(sd(z))||sd(z)<=.Machine$double.eps) {
    z0<-if(length(z)) mean(z) else 0
    return(function(xnew) rep(z0,length(xnew)))
  }
  zb<-tapply(z,x,mean); xu<-as.numeric(names(zb)); zu<-as.numeric(zb)
  ord<-order(xu); approxfun(xu[ord],zu[ord],rule=2)
}
make_fold_id <- function(n,k=2L,seed=NULL) {
  if(!is.null(seed)) set.seed(seed)
  sample(rep(seq_len(k),length.out=n),n,replace=FALSE)
}
crossfit_metrics <- function(x,y,fold_id=NULL,seed=NULL,standardize_x=FALSE,span=ACE_SPAN) {
  x<-as.matrix(x); y<-as.numeric(y)
  if(is.null(fold_id)) fold_id<-make_fold_id(nrow(x),CF_FOLDS,seed)
  rows<-list()
  for(f in sort(unique(fold_id))) {
    te<-which(fold_id==f); tr<-which(fold_id!=f)
    xtr<-x[tr,,drop=FALSE]; xte<-x[te,,drop=FALSE]
    if(standardize_x) {
      st<-standardize_train_test(xtr,xte); xtr<-st$train; xte<-st$test
    }
    ytr<-y[tr]; yte<-y[te]
    R0<-distance_correlation(xte,yte)
    fitlm<-lm(y~.,data=data.frame(y=ytr,xtr,check.names=FALSE))
    pred<-as.numeric(predict(fitlm,newdata=data.frame(xte,check.names=FALSE)))
    R1<-distance_correlation(pred,yte)
    af<-fit_ace_safely(xtr,ytr,span)
    if(!isTRUE(af$success)) {
      rows[[length(rows)+1L]]<-data.frame(R0=R0,R1=R1,R2=NA,R3=NA,GP=NA,GR=NA,success=FALSE)
      next
    }
    txte<-matrix(NA_real_,nrow(xte),ncol(xte))
    for(j in seq_len(ncol(xte))) {
      fj<-make_interp_function(xtr[,j],af$tx[,j]); txte[,j]<-fj(xte[,j])
    }
    tf<-make_interp_function(ytr,af$ty); tyte<-tf(yte); score<-rowSums(txte)
    R2<-distance_correlation(score,yte); R3<-distance_correlation(score,tyte)
    rows[[length(rows)+1L]]<-data.frame(
      R0=R0,R1=R1,R2=R2,R3=R3,GP=R2-R1,GR=R3-R2,success=TRUE
    )
  }
  tab<-do.call(rbind,rows)
  list(
    success=all(tab$success),R0=safe_mean(tab$R0),R1=safe_mean(tab$R1),
    R2=safe_mean(tab$R2),R3=safe_mean(tab$R3),
    GP=safe_mean(tab$GP),GR=safe_mean(tab$GR),fold_id=fold_id
  )
}

h2<-function(x)(x^2-1)/sqrt(2)
h3<-function(x)(x^3-3*x)/sqrt(6)
h_abs<-function(x)(abs(x)-sqrt(2/pi))/sqrt(1-2/pi)
h_sin<-function(x) sin(x)/sqrt((1-exp(-2))/2)
h_cos<-function(x)(cos(x)-exp(-.5))/sqrt((1+exp(-2))/2-exp(-1))
heterogeneous_component<-function(x,j) {
  switch(as.character(((j-1L)%%5L)+1L),
         "1"=h2(x),"2"=h3(x),"3"=h_sin(x),"4"=h_abs(x),"5"=h_cos(x))
}
raw_extended_signal<-function(x,condition,s) {
  a<-x[,seq_len(s),drop=FALSE]
  if(condition=="P6_dense_quadratic_additive") return(rowSums(apply(a,2,h2))/sqrt(s))
  if(condition=="P7_dense_heterogeneous_additive") {
    m<-sapply(seq_len(s),function(j) heterogeneous_component(a[,j],j))
    return(rowSums(m)/sqrt(s))
  }
  if(condition=="P8_chain_pairwise_interaction") {
    z<-rep(0,nrow(x)); for(j in seq_len(s-1L)) z<-z+a[,j]*a[,j+1L]
    return(z/sqrt(s-1L))
  }
  if(condition=="P9_quadratic_single_index") return(h2(rowSums(a)/sqrt(s)))
  if(condition=="P10_threshold_additive") return(rowSums(ifelse(a>0,1,-1))/sqrt(s))
  if(condition=="P11_mixed_main_interaction") {
    A<-rowSums(apply(a,2,h2))/sqrt(s)
    I<-rep(0,nrow(x)); for(j in seq_len(s-1L)) I<-I+a[,j]*a[,j+1L]
    return((A+I/sqrt(s-1L))/sqrt(2))
  }
  if(condition=="P12_highorder_sign_interaction") return(apply(ifelse(a>0,1,-1),1,prod))
  stop("Unknown condition")
}
generate_extended_data_exact<-function(condition,n,p,s,target_r2,seed) {
  set.seed(seed)
  x<-matrix(rnorm(n*p),n,p); colnames(x)<-paste0("X",seq_len(p))
  eta<-raw_extended_signal(x,condition,s)
  y<-eta+r2_to_noise_sd(target_r2)*rnorm(n)
  list(x=x,eta=eta,y=y,seed=seed)
}
crossfit_oracle_dcor<-function(eta,y,fold_id) {
  mean(vapply(sort(unique(fold_id)),function(f) {
    te<-which(fold_id==f); distance_correlation(eta[te],y[te])
  },numeric(1)))
}

run_checkpointed_dataset_block<-function(ids,worker,checkpoint,label,batch_size=10L) {
  old<-if(file.exists(checkpoint)) tryCatch(readRDS(checkpoint),error=function(e)NULL) else NULL
  done<-if(!is.null(old)&&"dataset_id"%in%names(old)) unique(old$dataset_id) else integer(0)
  rem<-setdiff(ids,done)
  if(!length(rem)) return(old)
  chunks<-split(rem,ceiling(seq_along(rem)/batch_size)); out<-old
  for(i in seq_along(chunks)) {
    part<-future.apply::future_lapply(chunks[[i]],worker,future.seed=TRUE,future.scheduling=1)
    out<-rbind(out,do.call(rbind,part)); saveRDS(out,checkpoint)
    log_message("[",label,"] batch ",i,"/",length(chunks),
                " | completed ",length(unique(out$dataset_id)),"/",length(ids))
  }
  out
}

EXT_SUMMARY_FILE<-file.path(EXT_SIM_DIR,"F_extended_P6_P12_summary.csv")
EXT_RAW_RDS<-file.path(EXT_SIM_DIR,"F_extended_P6_P12_replications.rds")
EXT_RAW_CSV<-file.path(EXT_SIM_DIR,"F_extended_P6_P12_replications.csv")
if(!file.exists(EXT_SUMMARY_FILE)) stop("Missing: ",EXT_SUMMARY_FILE)
extended_summary<-read.csv(EXT_SUMMARY_FILE,stringsAsFactors=FALSE,check.names=FALSE)
extended_raw<-if(file.exists(EXT_RAW_RDS)) readRDS(EXT_RAW_RDS) else
  if(file.exists(EXT_RAW_CSV)) read.csv(EXT_RAW_CSV,stringsAsFactors=FALSE,check.names=FALSE) else NULL


# =============================================================================
# Track A: publication-ready replot from existing P6--P12 summary
# =============================================================================
make_publication_figures <- function(summary) {
  d<-summary[summary$n==500 & summary$p==10 & abs(summary$target_r2-.5)<1e-12,,drop=FALSE]

  pdf(file.path(SUPP_FIGURES,"Figure_S1_extended_active_dimension_publication.pdf"),
      width=10,height=4.8)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,2),mar=c(4.3,4.5,2,1))

  draw_metric<-function(metric,ylab,title) {
    yr<-range(c(0,d[[metric]]),finite=TRUE); pad<-max(.01,diff(yr)*.08)
    plot(NA,xlim=c(1.5,10.5),ylim=c(yr[1]-pad,yr[2]+pad),xaxt="n",
         xlab="Number of active predictors (s)",ylab=ylab,main=title)
    axis(1,at=c(2,5,10)); grid(col="grey90"); abline(h=0,lty=2,col="grey55")
    for(cond in extended_condition_levels) {
      z<-d[d$condition==cond,,drop=FALSE]; z<-z[order(z$s),]
      lines(z$s,z[[metric]],type="b",lwd=2,
            pch=scenario_point_shapes[cond],lty=scenario_line_types[cond],
            col=scenario_colors[cond])
    }
    legend("topright",legend=extended_condition_ids[extended_condition_levels],
           col=scenario_colors[extended_condition_levels],
           pch=scenario_point_shapes[extended_condition_levels],
           lty=scenario_line_types[extended_condition_levels],lwd=2,bty="n",cex=.78)
  }
  draw_metric("cf_GP_mean",expression(paste("Mean cross-fitted ",G[P])),
              expression(paste("(a) Predictor-side gain ",G[P])))
  draw_metric("cf_GR_mean",expression(paste("Mean cross-fitted ",G[R])),
              expression(paste("(b) Response-transformation gain ",G[R])))
  dev.off(); par(oldpar)

  pdf(file.path(SUPP_FIGURES,"Figure_S2_extended_GP_GR_map_by_s_publication.pdf"),
      width=12,height=4.2)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,3),mar=c(4.2,4.3,2,1))
  xr<-range(c(0,d$cf_GP_mean),finite=TRUE); yr<-range(c(0,d$cf_GR_mean),finite=TRUE)
  xpad<-max(.015,diff(xr)*.08); ypad<-max(.008,diff(yr)*.10)
  for(s0 in c(2,5,10)) {
    z<-d[d$s==s0,,drop=FALSE]
    plot(z$cf_GP_mean,z$cf_GR_mean,
         pch=scenario_point_shapes[z$condition],col=scenario_colors[z$condition],cex=1.25,
         xlim=c(xr[1]-xpad,xr[2]+xpad),ylim=c(yr[1]-ypad,yr[2]+ypad),
         xlab=expression(paste("Cross-fitted ",G[P])),
         ylab=expression(paste("Cross-fitted ",G[R])),
         main=paste0("Active dimension s = ",s0))
    grid(col="grey90"); abline(h=0,v=0,lty=2,col="grey55")
    text(z$cf_GP_mean,z$cf_GR_mean,labels=z$condition_id,pos=3,cex=.78,
         col=scenario_colors[z$condition])
  }
  dev.off(); par(oldpar)

  write.csv(d,file.path(SUPP_TABLES,"Table_S1_extended_representative_n500_p10_R2_05.csv"),
            row.names=FALSE,fileEncoding="UTF-8")
  log_message("[Track A] publication figures complete.")
}
if(RUN_REPLOT_EXISTING) make_publication_figures(extended_summary)


# =============================================================================
# Track B: oracle recovery using same stored seeds
# =============================================================================
run_oracle_track <- function() {
  if(is.null(extended_raw)) stop("Oracle track needs F_extended_P6_P12_replications.rds/csv")
  sub<-extended_raw[
    extended_raw$n==500 & extended_raw$p==10 &
      abs(extended_raw$target_r2-.5)<1e-12 &
      extended_raw$replication<=ORACLE_REPS_PER_CELL,,drop=FALSE
  ]
  checkpoint<-file.path(SUPP_CHECKPOINT,"oracle_P6_P12.rds")
  if(file.exists(checkpoint)) {
    raw<-readRDS(checkpoint)
  } else {
    rows<-split(sub,seq_len(nrow(sub)))
    ans<-future.apply::future_lapply(rows,function(r) {
      r<-r[1,]
      dat<-generate_extended_data_exact(
        r$condition,as.integer(r$n),as.integer(r$p),as.integer(r$s),
        as.numeric(r$target_r2),as.integer(r$seed)
      )
      fid<-make_fold_id(nrow(dat$x),CF_FOLDS,as.integer(r$seed)+500000L)
      oracle<-crossfit_oracle_dcor(dat$eta,dat$y,fid)
      data.frame(
        condition=r$condition,condition_id=r$condition_id,s=r$s,
        replication=r$replication,seed=r$seed,
        cf_R2=r$cf_R2,cf_R3=r$cf_R3,cf_GP=r$cf_GP,cf_GR=r$cf_GR,
        oracle_cf_dCor_etaY=oracle,
        recovery_fraction_R2=r$cf_R2/oracle,
        recovery_fraction_R3=r$cf_R3/oracle,
        oracle_gap_R2=oracle-r$cf_R2,
        oracle_gap_R3=oracle-r$cf_R3,
        stringsAsFactors=FALSE
      )
    },future.seed=TRUE,future.scheduling=2)
    raw<-do.call(rbind,ans); saveRDS(raw,checkpoint)
  }

  keys<-unique(raw[,c("condition","condition_id","s")])
  summ<-do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
    k<-keys[i,]; z<-raw[raw$condition==k$condition & raw$s==k$s,,drop=FALSE]
    data.frame(
      condition=k$condition,condition_id=k$condition_id,s=k$s,replications=nrow(z),
      cf_R2_mean=safe_mean(z$cf_R2),cf_R3_mean=safe_mean(z$cf_R3),
      cf_GP_mean=safe_mean(z$cf_GP),cf_GR_mean=safe_mean(z$cf_GR),
      oracle_mean=safe_mean(z$oracle_cf_dCor_etaY),
      oracle_sd=safe_sd(z$oracle_cf_dCor_etaY),
      recovery_R2_mean=safe_mean(z$recovery_fraction_R2),
      recovery_R3_mean=safe_mean(z$recovery_fraction_R3),
      gap_R2_mean=safe_mean(z$oracle_gap_R2),gap_R3_mean=safe_mean(z$oracle_gap_R3),
      stringsAsFactors=FALSE
    )
  }))
  write.csv(raw,file.path(SUPP_TABLES,"oracle_P6_P12_raw.csv"),row.names=FALSE)
  write.csv(summ,file.path(SUPP_TABLES,"oracle_P6_P12_summary.csv"),row.names=FALSE)

  pdf(file.path(SUPP_FIGURES,"Figure_S3_oracle_recovery_P6_P12.pdf"),width=10,height=4.8)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,2),mar=c(4.3,4.5,2,1))
  plot(NA,xlim=c(1.5,10.5),ylim=range(c(0,summ$oracle_mean),finite=TRUE),xaxt="n",
       xlab="Number of active predictors (s)",
       ylab=expression(paste("Oracle cross-fitted dCor(",eta(X),", Y)")),
       main="(a) Underlying oracle dependence")
  axis(1,at=c(2,5,10)); grid(col="grey90")
  for(cond in extended_condition_levels) {
    z<-summ[summ$condition==cond,,drop=FALSE]; z<-z[order(z$s),]
    lines(z$s,z$oracle_mean,type="b",lwd=2,col=scenario_colors[cond],
          pch=scenario_point_shapes[cond],lty=scenario_line_types[cond])
  }
  legend("topright",legend=extended_condition_ids[extended_condition_levels],
         col=scenario_colors[extended_condition_levels],
         pch=scenario_point_shapes[extended_condition_levels],
         lty=scenario_line_types[extended_condition_levels],lwd=2,bty="n",cex=.78)

  plot(NA,xlim=c(1.5,10.5),ylim=range(c(0,1,summ$recovery_R3_mean),finite=TRUE),xaxt="n",
       xlab="Number of active predictors (s)",
       ylab=expression(paste("Recovery fraction: ",R[3]," / oracle dCor")),
       main="(b) Dependence recovered by stagewise ACE")
  axis(1,at=c(2,5,10)); grid(col="grey90"); abline(h=1,lty=2,col="grey55")
  for(cond in extended_condition_levels) {
    z<-summ[summ$condition==cond,,drop=FALSE]; z<-z[order(z$s),]
    lines(z$s,z$recovery_R3_mean,type="b",lwd=2,col=scenario_colors[cond],
          pch=scenario_point_shapes[cond],lty=scenario_line_types[cond])
  }
  dev.off(); par(oldpar)
  log_message("[Track B] oracle complete.")
  list(raw=raw,summary=summ)
}
oracle_track<-if(RUN_ORACLE_DIAGNOSTIC) run_oracle_track() else NULL


# =============================================================================
# Track C: representative robustness
# =============================================================================
ROBUST_SCENARIOS<-c(
  "P6_dense_quadratic_additive","P8_chain_pairwise_interaction",
  "P11_mixed_main_interaction","P12_highorder_sign_interaction"
)
ROBUST_DISTRIBUTIONS<-c("normal","t5","uniform_gaussian_copula")
ROBUST_RHO<-c(0,.5)
ROBUST_N<-500L; ROBUST_P<-10L; ROBUST_S<-5L; ROBUST_R2<-.5

ar1_cor<-function(p,rho) outer(seq_len(p),seq_len(p),function(i,j)rho^abs(i-j))
make_x_robust<-function(n,p,distribution,rho,seed) {
  set.seed(seed); z<-matrix(rnorm(n*p),n,p)%*%chol(ar1_cor(p,rho))
  if(distribution=="normal") x<-z
  else if(distribution=="t5") x<-z/sqrt(rchisq(n,5)/5)*sqrt(3/5)
  else if(distribution=="uniform_gaussian_copula") x<-(2*pnorm(z)-1)*sqrt(3)
  else stop("Unknown distribution")
  colnames(x)<-paste0("X",seq_len(p)); x
}

calibrate_robust_signals<-function() {
  g<-expand.grid(condition=ROBUST_SCENARIOS,distribution=ROBUST_DISTRIBUTIONS,
                 rho=ROBUST_RHO,stringsAsFactors=FALSE)
  out<-do.call(rbind,lapply(seq_len(nrow(g)),function(i) {
    x<-make_x_robust(ROBUST_CALIBRATION_N,ROBUST_P,g$distribution[i],g$rho[i],
                     BASE_SEED+20000000L+i*1000L)
    v<-raw_extended_signal(x,g$condition[i],ROBUST_S)
    data.frame(condition=g$condition[i],condition_id=extended_condition_ids[g$condition[i]],
               distribution=g$distribution[i],rho=g$rho[i],
               center=mean(v),scale=sd(v),stringsAsFactors=FALSE)
  }))
  write.csv(out,file.path(SUPP_TABLES,"robustness_signal_calibration.csv"),row.names=FALSE)
  out
}

run_robustness_track<-function() {
  cal<-calibrate_robust_signals(); all<-list(); idx<-1L
  for(cond in ROBUST_SCENARIOS) for(dist0 in ROBUST_DISTRIBUTIONS) for(rho0 in ROBUST_RHO) {
    cc<-cal[cal$condition==cond & cal$distribution==dist0 & abs(cal$rho-rho0)<1e-12,,drop=FALSE]
    cp<-file.path(SUPP_CHECKPOINT,paste0("robust_",extended_condition_ids[cond],"_",dist0,
                                        "_rho",gsub("\\.","",rho0),".rds"))
    ans<-run_checkpointed_dataset_block(seq_len(ROBUST_B),function(id) {
      ci<-match(cond,ROBUST_SCENARIOS); di<-match(dist0,ROBUST_DISTRIBUTIONS); ri<-match(rho0,ROBUST_RHO)
      seed<-BASE_SEED+30000000L+ci*1000000L+di*100000L+ri*10000L+id
      x<-make_x_robust(ROBUST_N,ROBUST_P,dist0,rho0,seed)
      rawsig<-raw_extended_signal(x,cond,ROBUST_S)
      eta<-(rawsig-cc$center)/cc$scale
      y<-eta+r2_to_noise_sd(ROBUST_R2)*rnorm(ROBUST_N)
      fid<-make_fold_id(ROBUST_N,CF_FOLDS,seed+500000L)
      cf<-crossfit_metrics(x,y,fold_id=fid)
      data.frame(
        dataset_id=id,condition=cond,condition_id=extended_condition_ids[cond],
        distribution=dist0,rho=rho0,cf_R0=cf$R0,cf_R1=cf$R1,cf_R2=cf$R2,cf_R3=cf$R3,
        cf_GP=cf$GP,cf_GR=cf$GR,oracle_cf_dCor_etaY=crossfit_oracle_dcor(eta,y,fid),
        cf_success=cf$success,stringsAsFactors=FALSE
      )
    },cp,paste0("Robust ",extended_condition_ids[cond]," ",dist0," rho=",rho0),20L)
    all[[idx]]<-ans; idx<-idx+1L
  }
  raw<-do.call(rbind,all)
  keys<-unique(raw[,c("condition","condition_id","distribution","rho")])
  summ<-do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
    k<-keys[i,]
    z<-raw[raw$condition==k$condition & raw$distribution==k$distribution &
             abs(raw$rho-k$rho)<1e-12,,drop=FALSE]
    data.frame(condition=k$condition,condition_id=k$condition_id,
               distribution=k$distribution,rho=k$rho,replications=nrow(z),
               cf_GP_mean=safe_mean(z$cf_GP),cf_GP_sd=safe_sd(z$cf_GP),
               cf_GR_mean=safe_mean(z$cf_GR),cf_GR_sd=safe_sd(z$cf_GR),
               cf_R3_mean=safe_mean(z$cf_R3),oracle_mean=safe_mean(z$oracle_cf_dCor_etaY),
               success_rate=mean(z$cf_success),stringsAsFactors=FALSE)
  }))
  write.csv(raw,file.path(SUPP_TABLES,"robustness_P6_P8_P11_P12_raw.csv"),row.names=FALSE)
  write.csv(summ,file.path(SUPP_TABLES,"robustness_P6_P8_P11_P12_summary.csv"),row.names=FALSE)

  draw_robust<-function(metric,ylab,file) {
    pdf(file.path(SUPP_FIGURES,file),width=9,height=7)
    oldpar<-par(no.readonly=TRUE); par(mfrow=c(2,2),mar=c(4.4,4.5,2.4,1))
    labs<-c(normal="Normal",t5="t(5)",uniform_gaussian_copula="Uniform")
    for(cond in ROBUST_SCENARIOS) {
      z<-summ[summ$condition==cond,,drop=FALSE]
      yr<-range(c(0,z[[metric]]),finite=TRUE); pad<-max(.01,diff(yr)*.1)
      plot(NA,xlim=c(.7,3.3),ylim=c(yr[1]-pad,yr[2]+pad),xaxt="n",
           xlab="Predictor distribution",ylab=ylab,main=extended_condition_ids[cond])
      axis(1,at=1:3,labels=labs[ROBUST_DISTRIBUTIONS]); grid(col="grey90"); abline(h=0,lty=2,col="grey55")
      for(r in ROBUST_RHO) {
        zz<-z[abs(z$rho-r)<1e-12,,drop=FALSE]
        zz<-zz[match(ROBUST_DISTRIBUTIONS,zz$distribution),]
        lines(1:3,zz[[metric]],type="b",lwd=2,col=scenario_colors[cond],
              pch=if(r==0)16 else 17,lty=if(r==0)1 else 2)
      }
      legend("topright",legend=c(expression(rho==0),expression(rho==0.5)),
             pch=c(16,17),lty=c(1,2),lwd=2,col=scenario_colors[cond],bty="n",cex=.8)
    }
    dev.off(); par(oldpar)
  }
  draw_robust("cf_GP_mean",expression(paste("Mean cross-fitted ",G[P])),"Figure_S4_robustness_GP.pdf")
  draw_robust("cf_GR_mean",expression(paste("Mean cross-fitted ",G[R])),"Figure_S5_robustness_GR.pdf")
  log_message("[Track C] robustness complete.")
  list(raw=raw,summary=summ)
}
robustness_track<-if(RUN_ROBUSTNESS) run_robustness_track() else NULL


# =============================================================================
# Bootstrap helpers
# =============================================================================
fit_linear_null<-function(x,y) {
  fit<-lm(y~.,data=data.frame(y=y,x,check.names=FALSE))
  r<-as.numeric(residuals(fit)); r<-r-mean(r)
  h<-as.numeric(hatvalues(fit))
  hc2<-as.numeric(residuals(fit))/sqrt(pmax(1-h,1e-8)); hc2<-hc2-mean(hc2)
  list(fitted=as.numeric(fitted(fit)),residual=r,residual_hc2=hc2)
}
mammen_multiplier<-function(n) {
  s5<-sqrt(5); low<-(1-s5)/2; high<-(1+s5)/2; p_low<-(s5+1)/(2*s5)
  ifelse(runif(n)<p_low,low,high)
}
make_bootstrap_y<-function(null_fit,method,seed) {
  set.seed(seed); n<-length(null_fit$fitted)
  if(method=="residual") {
    e<-sample(null_fit$residual,n,replace=TRUE)
  } else if(method=="wild_rademacher_hc2") {
    e<-null_fit$residual_hc2*sample(c(-1,1),n,replace=TRUE)
  } else if(method=="wild_mammen_hc2") {
    e<-null_fit$residual_hc2*mammen_multiplier(n)
  } else stop("Unknown bootstrap method")
  null_fit$fitted+e
}
plus_one_p<-function(v,obs) {
  v<-v[is.finite(v)]
  if(!length(v)||!is.finite(obs)) return(NA_real_)
  (1+sum(v>=obs))/(1+length(v))
}
bootstrap_one_dataset<-function(x,y,fold_id,methods,B,seed_base,standardize_x=FALSE) {
  observed<-crossfit_metrics(x,y,fold_id=fold_id,standardize_x=standardize_x)
  nf<-fit_linear_null(x,y); out<-list()
  method_levels<-c("residual","wild_rademacher_hc2","wild_mammen_hc2")
  for(method in methods) {
    gp<-gr<-rep(NA_real_,B)
    for(b in seq_len(B)) {
      ys<-make_bootstrap_y(nf,method,seed_base+match(method,method_levels)*1000000L+b)
      cf<-crossfit_metrics(x,ys,fold_id=fold_id,standardize_x=standardize_x)
      gp[b]<-cf$GP; gr[b]<-cf$GR
    }
    out[[method]]<-data.frame(
      method=method,observed_GP=observed$GP,observed_GR=observed$GR,
      null_GP_mean=safe_mean(gp),null_GP_q95=safe_quantile(gp,.95),
      null_GR_mean=safe_mean(gr),null_GR_q95=safe_quantile(gr,.95),
      p_GP=plus_one_p(gp,observed$GP),p_GR=plus_one_p(gr,observed$GR),
      n_valid_GP=sum(is.finite(gp)),n_valid_GR=sum(is.finite(gr)),
      stringsAsFactors=FALSE
    )
  }
  do.call(rbind,out)
}


# =============================================================================
# Track D: null-bootstrap calibration
# =============================================================================
NULL_N_VALUES<-c(200L,500L,1000L); NULL_P<-10L; NULL_S<-5L; NULL_R2<-.5
NULL_ERROR_TYPES<-c("normal","centered_exponential","heteroskedastic_normal")

generate_linear_null_data<-function(n,error_type,dataset_id) {
  ni<-match(n,NULL_N_VALUES); ei<-match(error_type,NULL_ERROR_TYPES)
  seed<-BASE_SEED+40000000L+ni*1000000L+ei*100000L+dataset_id
  set.seed(seed)
  x<-matrix(rnorm(n*NULL_P),n,NULL_P); colnames(x)<-paste0("X",seq_len(NULL_P))
  eta<-rowSums(x[,seq_len(NULL_S),drop=FALSE])/sqrt(NULL_S)
  if(error_type=="normal") eps<-rnorm(n)
  else if(error_type=="centered_exponential") eps<-rexp(n)-1
  else if(error_type=="heteroskedastic_normal") {
    sigma_raw<-.5+.75*abs(x[,1])
    sigma_scale<-sqrt(.25+.75*sqrt(2/pi)+.5625)
    eps<-(sigma_raw/sigma_scale)*rnorm(n)
  } else stop("Unknown null error")
  list(x=x,y=eta+r2_to_noise_sd(NULL_R2)*eps,eta=eta,seed=seed)
}

run_null_bootstrap_track<-function() {
  all<-list(); idx<-1L
  for(n0 in NULL_N_VALUES) for(err in NULL_ERROR_TYPES) {
    cp<-file.path(SUPP_CHECKPOINT,paste0("null_boot_n",n0,"_",err,".rds"))
    ans<-run_checkpointed_dataset_block(seq_len(NULL_BOOT_DATASETS),function(id) {
      dat<-generate_linear_null_data(n0,err,id)
      fid<-make_fold_id(n0,CF_FOLDS,dat$seed+500000L)
      bs<-bootstrap_one_dataset(dat$x,dat$y,fid,NULL_BOOT_METHODS,NULL_BOOT_B,
                                dat$seed+10000000L,FALSE)
      bs$dataset_id<-id; bs$n<-n0; bs$error_type<-err; bs$seed<-dat$seed
      bs$reject_GP_05<-is.finite(bs$p_GP)&bs$p_GP<=.05
      bs$reject_GR_05<-is.finite(bs$p_GR)&bs$p_GR<=.05
      bs
    },cp,paste0("Null boot n=",n0," ",err),5L)
    all[[idx]]<-ans; idx<-idx+1L
  }
  raw<-do.call(rbind,all)
  keys<-unique(raw[,c("n","error_type","method")])
  summ<-do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
    k<-keys[i,]; z<-raw[raw$n==k$n & raw$error_type==k$error_type & raw$method==k$method,,drop=FALSE]
    nd<-length(unique(z$dataset_id)); pg<-mean(z$reject_GP_05,na.rm=TRUE); pr<-mean(z$reject_GR_05,na.rm=TRUE)
    data.frame(
      n=k$n,error_type=k$error_type,method=k$method,datasets=nd,
      rejection_rate_GP=pg,mcse_GP=sqrt(pg*(1-pg)/nd),
      rejection_rate_GR=pr,mcse_GR=sqrt(pr*(1-pr)/nd),
      mean_p_GP=safe_mean(z$p_GP),mean_p_GR=safe_mean(z$p_GR),
      mean_observed_GP=safe_mean(z$observed_GP),mean_observed_GR=safe_mean(z$observed_GR),
      stringsAsFactors=FALSE
    )
  }))
  write.csv(raw,file.path(SUPP_TABLES,"bootstrap_null_calibration_raw.csv"),row.names=FALSE)
  write.csv(summ,file.path(SUPP_TABLES,"bootstrap_null_calibration_summary.csv"),row.names=FALSE)

  pdf(file.path(SUPP_FIGURES,"Figure_S6_null_bootstrap_calibration.pdf"),width=10,height=4.8)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,2),mar=c(4.3,4.5,2.1,1))
  ecol<-c(normal="#1B9E77",centered_exponential="#D95F02",heteroskedastic_normal="#7570B3")
  mlty<-c(residual=1,wild_rademacher_hc2=2); mpch<-c(residual=16,wild_rademacher_hc2=17)
  draw<-function(metric,title) {
    plot(NA,xlim=range(NULL_N_VALUES),ylim=c(0,max(.12,summ[[metric]],na.rm=TRUE)),
         xaxt="n",xlab="Sample size n",ylab="Empirical rejection rate",main=title)
    axis(1,at=NULL_N_VALUES); grid(col="grey90"); abline(h=.05,lty=3,lwd=1.5,col="grey45")
    for(err in NULL_ERROR_TYPES) for(m in NULL_BOOT_METHODS) {
      z<-summ[summ$error_type==err & summ$method==m,,drop=FALSE]; z<-z[order(z$n),]
      lines(z$n,z[[metric]],type="b",col=ecol[err],lty=mlty[m],pch=mpch[m],lwd=2)
    }
    legend("topright",
           legend=c("Normal / residual","Normal / wild HC2",
                    "Centered exp. / residual","Centered exp. / wild HC2",
                    "Heterosk. normal / residual","Heterosk. normal / wild HC2"),
           col=rep(ecol,each=2),lty=rep(c(1,2),3),pch=rep(c(16,17),3),lwd=2,bty="n",cex=.68)
  }
  draw("rejection_rate_GP",expression(paste("(a) ",G[P])))
  draw("rejection_rate_GR",expression(paste("(b) ",G[R])))
  dev.off(); par(oldpar)
  log_message("[Track D] null bootstrap complete.")
  list(raw=raw,summary=summ)
}
null_boot_track<-if(RUN_NULL_BOOTSTRAP_CALIBRATION) run_null_bootstrap_track() else NULL


# =============================================================================
# Track E: representative bootstrap power
# =============================================================================
POWER_SCENARIOS<-c(
  "P6_dense_quadratic_additive","P8_chain_pairwise_interaction",
  "P11_mixed_main_interaction","P12_highorder_sign_interaction"
)
POWER_N_VALUES<-c(200L,500L,1000L); POWER_P<-10L; POWER_S<-5L; POWER_R2<-.5

run_power_bootstrap_track<-function() {
  all<-list(); idx<-1L
  for(cond in POWER_SCENARIOS) for(n0 in POWER_N_VALUES) {
    cp<-file.path(SUPP_CHECKPOINT,paste0("power_boot_",extended_condition_ids[cond],"_n",n0,".rds"))
    ans<-run_checkpointed_dataset_block(seq_len(POWER_BOOT_DATASETS),function(id) {
      ci<-match(cond,POWER_SCENARIOS); ni<-match(n0,POWER_N_VALUES)
      seed<-BASE_SEED+50000000L+ci*1000000L+ni*100000L+id
      dat<-generate_extended_data_exact(cond,n0,POWER_P,POWER_S,POWER_R2,seed)
      fid<-make_fold_id(n0,CF_FOLDS,seed+500000L)
      bs<-bootstrap_one_dataset(dat$x,dat$y,fid,POWER_BOOT_METHODS,POWER_BOOT_B,
                                seed+10000000L,FALSE)
      bs$dataset_id<-id; bs$condition<-cond; bs$condition_id<-extended_condition_ids[cond]
      bs$n<-n0; bs$seed<-seed
      bs$reject_GP_05<-is.finite(bs$p_GP)&bs$p_GP<=.05
      bs$reject_GR_05<-is.finite(bs$p_GR)&bs$p_GR<=.05
      bs
    },cp,paste0("Power ",extended_condition_ids[cond]," n=",n0),5L)
    all[[idx]]<-ans; idx<-idx+1L
  }
  raw<-do.call(rbind,all)
  keys<-unique(raw[,c("condition","condition_id","n","method")])
  summ<-do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
    k<-keys[i,]; z<-raw[raw$condition==k$condition & raw$n==k$n & raw$method==k$method,,drop=FALSE]
    nd<-length(unique(z$dataset_id)); pg<-mean(z$reject_GP_05,na.rm=TRUE); pr<-mean(z$reject_GR_05,na.rm=TRUE)
    data.frame(
      condition=k$condition,condition_id=k$condition_id,n=k$n,method=k$method,datasets=nd,
      power_GP=pg,mcse_GP=sqrt(pg*(1-pg)/nd),power_GR=pr,mcse_GR=sqrt(pr*(1-pr)/nd),
      mean_observed_GP=safe_mean(z$observed_GP),mean_observed_GR=safe_mean(z$observed_GR),
      stringsAsFactors=FALSE
    )
  }))
  write.csv(raw,file.path(SUPP_TABLES,"bootstrap_extended_power_raw.csv"),row.names=FALSE)
  write.csv(summ,file.path(SUPP_TABLES,"bootstrap_extended_power_summary.csv"),row.names=FALSE)

  pdf(file.path(SUPP_FIGURES,"Figure_S7_extended_bootstrap_power.pdf"),width=10,height=4.8)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,2),mar=c(4.3,4.5,2.1,1))
  draw<-function(metric,title) {
    plot(NA,xlim=range(POWER_N_VALUES),ylim=c(0,1),xaxt="n",
         xlab="Sample size n",ylab="Empirical rejection rate",main=title)
    axis(1,at=POWER_N_VALUES); grid(col="grey90")
    for(cond in POWER_SCENARIOS) {
      z<-summ[summ$condition==cond,,drop=FALSE]; z<-z[order(z$n),]
      lines(z$n,z[[metric]],type="b",lwd=2,col=scenario_colors[cond],
            pch=scenario_point_shapes[cond],lty=scenario_line_types[cond])
    }
    legend("bottomright",legend=extended_condition_ids[POWER_SCENARIOS],
           col=scenario_colors[POWER_SCENARIOS],pch=scenario_point_shapes[POWER_SCENARIOS],
           lty=scenario_line_types[POWER_SCENARIOS],lwd=2,bty="n",cex=.8)
  }
  draw("power_GP",expression(paste("(a) ",G[P]," rejection rate")))
  draw("power_GR",expression(paste("(b) ",G[R]," rejection rate")))
  dev.off(); par(oldpar)
  log_message("[Track E] power bootstrap complete.")
  list(raw=raw,summary=summ)
}
power_boot_track<-if(RUN_EXTENDED_BOOTSTRAP_POWER) run_power_bootstrap_track() else NULL


# =============================================================================
# Track F: Concrete bootstrap refinement
# =============================================================================
CONCRETE_URL<-paste0(
  "https://archive.ics.uci.edu/static/public/165/",
  "concrete+compressive+strength.zip"
)

load_concrete_data<-function() {
  ddir<-file.path(SUPP_CONCRETE,"uci_data")
  if(!dir.exists(ddir)) dir.create(ddir,recursive=TRUE)
  zp<-file.path(ddir,"concrete_compressive_strength.zip")
  if(!file.exists(zp)) download.file(CONCRETE_URL,zp,mode="wb")
  uz<-file.path(ddir,"unzipped"); if(!dir.exists(uz)) dir.create(uz,recursive=TRUE)
  unzip(zp,exdir=uz,overwrite=TRUE)
  fs<-list.files(uz,recursive=TRUE,full.names=TRUE)
  xl<-fs[grepl("\\.xls[x]?$",fs,ignore.case=TRUE)]
  if(!length(xl)) stop("Concrete Excel file not found.")
  pref<-xl[grepl("Concrete_Data",basename(xl),ignore.case=TRUE)]
  xp<-if(length(pref)) pref[1] else xl[1]
  dat<-as.data.frame(readxl::read_excel(xp)); dat<-dat[complete.cases(dat),,drop=FALSE]
  x<-as.matrix(dat[,1:8]); y<-as.numeric(dat[,9])
  colnames(x)<-c("Cement","BlastFurnaceSlag","FlyAsh","Water","Superplasticizer",
                 "CoarseAggregate","FineAggregate","Age")
  list(x=x,y=y)
}

run_concrete_bootstrap_refinement<-function() {
  dat<-load_concrete_data(); x<-dat$x; y<-dat$y; n<-nrow(x)
  fid<-make_fold_id(n,CF_FOLDS,BASE_SEED+90000000L)
  observed<-crossfit_metrics(x,y,fold_id=fid,standardize_x=TRUE)
  nf<-fit_linear_null(x,y)
  all<-list()
  for(method in CONCRETE_BOOT_METHODS) {
    cp<-file.path(SUPP_CHECKPOINT,paste0("concrete_boot_",method,"_B",CONCRETE_BOOT_B,".rds"))
    if(file.exists(cp)) {
      vals<-readRDS(cp)
    } else {
      vals_list<-future.apply::future_lapply(seq_len(CONCRETE_BOOT_B),function(b) {
        ys<-make_bootstrap_y(
          nf,method,
          BASE_SEED+91000000L+match(method,CONCRETE_BOOT_METHODS)*1000000L+b
        )
        cf<-crossfit_metrics(x,ys,fold_id=fid,standardize_x=TRUE)
        data.frame(bootstrap_id=b,method=method,GP=cf$GP,GR=cf$GR,stringsAsFactors=FALSE)
      },future.seed=TRUE,future.scheduling=2)
      vals<-do.call(rbind,vals_list); saveRDS(vals,cp)
    }
    all[[method]]<-vals
  }
  raw<-do.call(rbind,all)
  summ<-do.call(rbind,lapply(CONCRETE_BOOT_METHODS,function(method) {
    z<-raw[raw$method==method,,drop=FALSE]
    data.frame(
      method=method,observed_GP=observed$GP,observed_GR=observed$GR,
      null_GP_mean=safe_mean(z$GP),null_GP_q95=safe_quantile(z$GP,.95),
      p_GP=plus_one_p(z$GP,observed$GP),
      null_GR_mean=safe_mean(z$GR),null_GR_q95=safe_quantile(z$GR,.95),
      p_GR=plus_one_p(z$GR,observed$GR),
      stringsAsFactors=FALSE
    )
  }))
  write.csv(raw,file.path(SUPP_CONCRETE,"concrete_bootstrap_refinement_raw.csv"),row.names=FALSE)
  write.csv(summ,file.path(SUPP_CONCRETE,"concrete_bootstrap_refinement_summary.csv"),row.names=FALSE)

  pdf(file.path(SUPP_FIGURES,"Figure_S8_concrete_bootstrap_method_sensitivity.pdf"),width=8,height=4.6)
  oldpar<-par(no.readonly=TRUE); par(mfrow=c(1,2),mar=c(5.5,4.5,2,1))
  labs<-c("Residual","Wild Rad.\nHC2","Wild Mammen\nHC2")
  barplot(summ$p_GP,names.arg=labs,ylim=c(0,max(.1,summ$p_GP,na.rm=TRUE)),
          ylab="One-sided p-value",main=expression(paste("(a) ",G[P])))
  abline(h=.05,lty=2,col="grey45")
  barplot(summ$p_GR,names.arg=labs,ylim=c(0,1),
          ylab="One-sided p-value",main=expression(paste("(b) ",G[R])))
  abline(h=.05,lty=2,col="grey45")
  dev.off(); par(oldpar)
  log_message("[Track F] Concrete bootstrap refinement complete.")
  list(raw=raw,summary=summ,observed=observed)
}
concrete_boot_track<-if(RUN_CONCRETE_BOOTSTRAP_REFINEMENT)
  run_concrete_bootstrap_refinement() else NULL


# =============================================================================
# Existing main-simulation Monte Carlo uncertainty table
# =============================================================================
make_mc_uncertainty_table<-function() {
  d<-extended_summary[
    extended_summary$n==500 & extended_summary$p==10 &
      abs(extended_summary$target_r2-.5)<1e-12,,drop=FALSE
  ]
  keep<-intersect(c(
    "condition","condition_id","n","p","s","target_r2","replications",
    "cf_GP_mean","cf_GP_sd","cf_GP_se","cf_GP_q025","cf_GP_median","cf_GP_q975",
    "cf_GR_mean","cf_GR_sd","cf_GR_se","cf_GR_q025","cf_GR_median","cf_GR_q975",
    "ace_success_rate","cf_success_rate"
  ),names(d))
  write.csv(d[,keep,drop=FALSE],
            file.path(SUPP_TABLES,"Table_S2_MC_uncertainty_extended_n500_p10_R2_05.csv"),
            row.names=FALSE)
}
make_mc_uncertainty_table()


# =============================================================================
# Workload / run metadata
# =============================================================================
workload<-data.frame(
  track=c("Oracle datasets","Robustness ACE/CF reps","Null bootstrap samples",
          "Extended-power bootstrap samples","Concrete bootstrap samples"),
  planned=c(
    7L*3L*ORACLE_REPS_PER_CELL,
    length(ROBUST_SCENARIOS)*length(ROBUST_DISTRIBUTIONS)*length(ROBUST_RHO)*ROBUST_B,
    length(NULL_N_VALUES)*length(NULL_ERROR_TYPES)*NULL_BOOT_DATASETS*NULL_BOOT_B*
      length(NULL_BOOT_METHODS),
    length(POWER_SCENARIOS)*length(POWER_N_VALUES)*POWER_BOOT_DATASETS*POWER_BOOT_B*
      length(POWER_BOOT_METHODS),
    CONCRETE_BOOT_B*length(CONCRETE_BOOT_METHODS)
  ),
  stringsAsFactors=FALSE
)
write.csv(workload,file.path(SUPP_ROOT,"WORKLOAD_PLAN.csv"),row.names=FALSE)

writeLines(c(
  paste0("completed_at=",format(Sys.time(),"%Y-%m-%d %H:%M:%S")),
  paste0("profile=",RUN_PROFILE),
  paste0("R_version=",R.version.string),
  paste0("acepack_version=",as.character(packageVersion("acepack"))),
  paste0("workers=",MAX_WORKERS),
  paste0("oracle_reps_per_cell=",ORACLE_REPS_PER_CELL),
  paste0("robust_B=",ROBUST_B),
  paste0("null_boot_datasets=",NULL_BOOT_DATASETS),
  paste0("null_boot_B=",NULL_BOOT_B),
  paste0("power_boot_datasets=",POWER_BOOT_DATASETS),
  paste0("power_boot_B=",POWER_BOOT_B),
  paste0("concrete_boot_B=",CONCRETE_BOOT_B),
  "",
  "Interpretation notes:",
  "GP and GR are diagnostic contrasts, not invariant structural parameters.",
  "Oracle dCor{eta(X),Y} distinguishes underlying signal from ACE-representation limits.",
  "Residual bootstrap is the baseline under homoskedastic exchangeable residuals.",
  "Wild Rademacher HC2 assesses heteroskedastic robustness.",
  "Concrete Mammen HC2 is an additional sensitivity analysis."
),file.path(SUPP_ROOT,"RUN_COMPLETED.txt"))

writeLines(capture.output(sessionInfo()),file.path(SUPP_ROOT,"sessionInfo.txt"))
log_message("============================================================")
log_message("Reviewer-oriented supplementary analyses completed.")
log_message("Output: ",normalizePath(SUPP_ROOT,winslash="/",mustWork=FALSE))
log_message("============================================================")

if(MAX_WORKERS>1L) future::plan(future::sequential)
