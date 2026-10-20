# ============================================================================
# PAPER MAIN ANALYSIS -- v4 ACEFIX + publication post-processing
# p-to-one ACE--distance correlation: simulation + UCI Concrete real data
#
# This script is the publication-scale successor to:
#   - p_to_one_pilot_v7.R
#   - real_data_concrete_v1.R
#
# Main scientific quantities
# --------------------------
# R0 = dCor(X, Y)                       : raw multivariate dependence
# R1 = dCor(S_linear, Y)                : linear-score representation
# R2 = dCor(S_ACE, Y)                   : ACE predictor-side representation
# R3 = dCor(S_ACE, theta_ACE(Y))        : full ACE representation
#
# GP = R2 - R1                          : predictor-side nonlinear gain
# GR = R3 - R2                          : response-transformation gain
#
# IMPORTANT INTERPRETATION
# ------------------------
# GP and GR are stagewise diagnostic contrasts. They are NOT an orthogonal
# variance decomposition and they are NOT causal effects.
#
# Publication-scale design implemented here
# ------------------------------------------
# Track A. Main factorial Monte Carlo
#   n in {200, 500, 1000}
#   p in {2, 5, 10, 20}
#   P0--P5 structural conditions
#   target oracle R^2 in {0.2, 0.5, 0.8} for P1--P5
#   B = 1000 replications per non-null cell; P0 is generated once per (n,p)
#   all primary metrics evaluated by 2-fold cross-fitting
#
# Track B. Linear-null inference experiment
#   n in {200, 500, 1000}, p = 5, target R^2 = 0.5
#   500 independent datasets per condition
#   499 residual-bootstrap samples per dataset
#   one-sided tests for GP and GR
#   P0/P1 act as size/null checks; P2--P5 act as power/structure checks
#
# Track C. Robustness
#   - correlated predictors: rho in {0, 0.5}, fixed signal variance
#   - ACE span: {0.3, 0.5, 0.7}
#   - error distributions: Gaussian, standardized t_3, centered exponential
#   - K-fold sensitivity: K in {2, 5}
#
# Track D. Method benchmark at representative setting
#   linear score vs ACE score vs GAM prediction vs random forest prediction
#
# Track E. UCI Concrete Compressive Strength data (Yeh, 1998)
#   - automated UCI download
#   - descriptive statistics + univariate dependence table
#   - full-sample and repeated cross-fitted R0--R3 / GP / GR
#   - 100 repeated 2-fold cross-fits
#   - residual-bootstrap and wild-bootstrap linear-null calibration (B=999 each)
#   - span and K-fold sensitivity
#   - ACE transformation plots and publication tables
#
# COMPUTATION
# -----------
# The default RUN_PROFILE = "paper_main" is intentionally large and may run for
# many hours. Checkpoints are written so an interrupted job can be resumed.
# On Windows/macOS/Linux, future.apply::multisession is used when workers > 1.
#
# NOTE ON n = 10,000
# ------------------
# The naive sample dCor implementation is O(n^2) in memory/time. Therefore
# n = 10,000 is NOT included in the automatic main run. Use it only in a
# separate computational/reference study with a memory-aware dCor implementation.
# ============================================================================

# ----------------------------------------------------------------------------
# 0. Packages
# ----------------------------------------------------------------------------
required_packages <- c("acepack", "readxl", "future", "future.apply", "mgcv", "ranger")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

# IMPORTANT:
# This analysis uses acepack::ace(..., control=list(span=..., maxit=...), on.error=...).
# In current acepack releases, set_control() is strict about the *class* of maxit.
# maxit must inherit from "numeric"; an R integer literal such as 50L is rejected.
# The helper make_ace_control() below explicitly converts span/maxit to double
# before every ACE call, and mandatory preflight tests run before long simulation.
MIN_ACEPACK_VERSION <- package_version("1.6.1")
ACEPACK_VERSION <- utils::packageVersion("acepack")
if (ACEPACK_VERSION < MIN_ACEPACK_VERSION) {
  stop(
    paste0(
      "acepack >= ", MIN_ACEPACK_VERSION, " is required, but installed version is ",
      ACEPACK_VERSION, ".\n",
      "Run install.packages('acepack'), restart R/RStudio, and source this script again.\n",
      "The long simulation will NOT start until this is fixed."
    )
  )
}

suppressPackageStartupMessages({
  library(acepack)
  library(readxl)
  library(future)
  library(future.apply)
})

# ----------------------------------------------------------------------------
# 1. Global configuration
# ----------------------------------------------------------------------------
RUN_PROFILE <- "paper_main"   # "smoke", "paper_fast", "paper_main"
BASE_SEED <- 20260826L
set.seed(BASE_SEED)

ACE_SPAN <- 0.5

# IMPORTANT (acepack >= 1.6.x):
# acepack::set_control() checks maxit with inherits(maxit, "numeric").
# Therefore an integer literal such as 50L is rejected even though
# is.numeric(50L) is TRUE. Keep ACE_MAXIT as a DOUBLE (50, not 50L).
ACE_MAXIT <- 50

CF_FOLDS <- 2L
ALPHA <- 0.05

# Main tracks
DO_REAL_DATA_FIRST <- TRUE
DO_MAIN_FACTORIAL <- TRUE
DO_LINEAR_NULL_INFERENCE <- TRUE
DO_CORRELATION_ROBUSTNESS <- TRUE
DO_SPAN_SENSITIVITY <- TRUE
DO_ERROR_ROBUSTNESS <- TRUE
DO_KFOLD_SENSITIVITY <- TRUE
DO_METHOD_BENCHMARK <- TRUE
DO_REAL_DATA <- TRUE

# Resume completed checkpoint blocks when the same profile is re-run.
RESUME_FROM_CHECKPOINTS <- TRUE

# For simulation, keep the v5--v7 scale convention exactly as designed.
# For real data, raw predictor units differ substantially, so fold-wise
# standardization is used before multivariate R0 calculation and model fitting.
SIM_STANDARDIZE_X <- FALSE
REAL_STANDARDIZE_X <- TRUE

# Parallel workers.
# Ryzen 7 8700F (8 physical cores / 16 threads) 연구실 PC 기준으로 8 workers를 요청한다.
# 다른 PC에서 실행하더라도 사용 가능한 논리 코어 수를 넘지 않도록 자동 제한한다.
REQUESTED_WORKERS <- 8L
available_cores <- tryCatch(future::availableCores(), error = function(e) 1L)
MAX_WORKERS <- max(1L, min(REQUESTED_WORKERS, max(1L, as.integer(available_cores) - 1L)))
if (!is.finite(MAX_WORKERS) || MAX_WORKERS < 1L) MAX_WORKERS <- 1L

# 각 R worker 내부에서 추가 스레드를 과도하게 생성하지 않도록 제한한다.
# (병렬 R process + 내부 BLAS/OpenMP 병렬화의 oversubscription 방지)
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")

if (MAX_WORKERS > 1L) {
  future::plan(future::multisession, workers = MAX_WORKERS)
} else {
  future::plan(future::sequential)
}
options(future.globals.maxSize = 4 * 1024^3)

# Console progress / ETA reporting.
SHOW_PROGRESS <- TRUE
# Main factorial tasks are relatively short, so larger batches reduce future overhead.
PROGRESS_BATCH_MAIN <- max(64L, MAX_WORKERS * 16L)
# Standard robustness/benchmark tasks.
PROGRESS_BATCH_STANDARD <- max(32L, MAX_WORKERS * 8L)
# Track B task = one dataset + many bootstrap refits; use one wave of workers per update.
PROGRESS_BATCH_INFERENCE <- max(1L, MAX_WORKERS)
# Real-data bootstrap tasks are lighter than Track B but still expensive.
PROGRESS_BATCH_REAL_BOOT <- max(16L, MAX_WORKERS * 4L)

# Output structure
# -------------------------------------------------------------------------
# 연구실 PC의 논문 본실험 작업 폴더를 명시적으로 고정한다.
# Windows 경로는 R에서 forward slash("/")를 사용하면 안전하다.
PROJECT_DIR <- Sys.getenv("PORTFOLIO_PROJECT_DIR", unset = getwd())

if (!dir.exists(PROJECT_DIR)) {
  dir.create(PROJECT_DIR, recursive = TRUE, showWarnings = FALSE)
}

# 이후 상대경로를 사용하는 코드가 있어도 항상 이 프로젝트 폴더 아래에서 실행되게 한다.
setwd(PROJECT_DIR)

# Use a new directory so no checkpoint from the earlier broken ACE-control run is reused.
OUTPUT_TAG <- "v6_acepack163_fixed"
ROOT_OUTPUT <- file.path(
  PROJECT_DIR,
  paste0("paper_main_output_", RUN_PROFILE, "_", OUTPUT_TAG)
)
SIM_OUTPUT <- file.path(ROOT_OUTPUT, "simulation")
REAL_OUTPUT <- file.path(ROOT_OUTPUT, "real_data_concrete")
CHECKPOINT_OUTPUT <- file.path(ROOT_OUTPUT, "checkpoints")

for (d in c(ROOT_OUTPUT, SIM_OUTPUT, REAL_OUTPUT, CHECKPOINT_OUTPUT)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

LOG_FILE <- file.path(ROOT_OUTPUT, "run_log.txt")

cat("\n============================================================\n")
cat("PROJECT DIRECTORY\n")
cat("============================================================\n")
cat("Project :", normalizePath(PROJECT_DIR, winslash = "/", mustWork = FALSE), "\n")
cat("Output  :", normalizePath(ROOT_OUTPUT, winslash = "/", mustWork = FALSE), "\n")
cat("Workers :", MAX_WORKERS, "\n")
cat("============================================================\n\n")

log_message <- function(...) {
  txt <- paste(..., collapse = "")
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", txt)
  cat(line, "\n")
  cat(line, "\n", file = LOG_FILE, append = TRUE)
}

# Profile-specific computational sizes
if (RUN_PROFILE == "smoke") {
  MAIN_N_VALUES <- c(200L)
  MAIN_P_VALUES <- c(2L, 5L)
  MAIN_R2_VALUES <- c(0.5)
  MAIN_B <- 3L

  INFER_N_VALUES <- c(200L)
  INFER_P <- 5L
  INFER_R2 <- 0.5
  INFER_REPLICATIONS <- 2L
  INFER_BOOT_B <- 5L

  ROBUST_N <- 200L
  ROBUST_P <- 5L
  ROBUST_R2 <- 0.5
  ROBUST_B <- 3L

  SPAN_B <- 3L
  ERROR_B <- 3L
  KFOLD_B <- 3L
  BENCHMARK_B <- 3L

  REAL_CF_REPEATS <- 3L
  REAL_BOOT_B <- 9L
  REAL_SENS_REPEATS <- 2L
  REAL_BENCH_REPEATS <- 2L

} else if (RUN_PROFILE == "paper_fast") {
  MAIN_N_VALUES <- c(200L, 500L, 1000L)
  MAIN_P_VALUES <- c(2L, 5L, 10L, 20L)
  MAIN_R2_VALUES <- c(0.2, 0.5, 0.8)
  MAIN_B <- 300L

  INFER_N_VALUES <- c(200L, 500L, 1000L)
  INFER_P <- 5L
  INFER_R2 <- 0.5
  INFER_REPLICATIONS <- 100L
  INFER_BOOT_B <- 99L

  ROBUST_N <- 500L
  ROBUST_P <- 5L
  ROBUST_R2 <- 0.5
  ROBUST_B <- 150L

  SPAN_B <- 150L
  ERROR_B <- 150L
  KFOLD_B <- 150L
  BENCHMARK_B <- 100L

  REAL_CF_REPEATS <- 50L
  REAL_BOOT_B <- 499L
  REAL_SENS_REPEATS <- 25L
  REAL_BENCH_REPEATS <- 25L

} else if (RUN_PROFILE == "paper_main") {
  # Final-paper scale.
  MAIN_N_VALUES <- c(200L, 500L, 1000L)
  MAIN_P_VALUES <- c(2L, 5L, 10L, 20L)
  MAIN_R2_VALUES <- c(0.2, 0.5, 0.8)
  MAIN_B <- 1000L

  # Linear-null inference: Bboot=499 gives p-value resolution 0.002.
  INFER_N_VALUES <- c(200L, 500L, 1000L)
  INFER_P <- 5L
  INFER_R2 <- 0.5
  INFER_REPLICATIONS <- 500L
  INFER_BOOT_B <- 499L

  ROBUST_N <- 500L
  ROBUST_P <- 5L
  ROBUST_R2 <- 0.5
  ROBUST_B <- 500L

  SPAN_B <- 500L
  ERROR_B <- 500L
  KFOLD_B <- 500L
  BENCHMARK_B <- 300L

  REAL_CF_REPEATS <- 100L
  REAL_BOOT_B <- 999L
  REAL_SENS_REPEATS <- 50L
  REAL_BENCH_REPEATS <- 50L

} else {
  stop("Unknown RUN_PROFILE: ", RUN_PROFILE)
}

RHO_VALUES <- c(0, 0.5)
SPAN_VALUES <- c(0.3, 0.5, 0.7)
ERROR_DISTRIBUTIONS <- c("normal", "t3", "centered_exponential")
KFOLD_VALUES <- c(2L, 5L)

log_message("Profile = ", RUN_PROFILE,
            "; workers = ", MAX_WORKERS,
            "; root output = ", normalizePath(ROOT_OUTPUT, winslash = "/", mustWork = FALSE))

# ----------------------------------------------------------------------------
# 2. Structural conditions
# ----------------------------------------------------------------------------
condition_levels <- c(
  "P0_null",
  "P1_linear",
  "P2_symmetric_additive",
  "P3_heterogeneous_additive",
  "P4_main_interaction",
  "P5_pure_interaction"
)

condition_ids <- c("P0", "P1", "P2", "P3", "P4", "P5")
names(condition_ids) <- condition_levels

condition_labels <- c(
  P0_null = "P0: independence",
  P1_linear = "P1: linear additive",
  P2_symmetric_additive = "P2: symmetric additive nonlinear",
  P3_heterogeneous_additive = "P3: heterogeneous additive nonlinear",
  P4_main_interaction = "P4: main effect + interaction",
  P5_pure_interaction = "P5: pure interaction"
)

condition_formulas <- c(
  P0_null = "eta=0; Y=epsilon",
  P1_linear = "eta=(X1+X2)/sqrt(2)",
  P2_symmetric_additive = "eta=(X1^2+X2^2-2)/2",
  P3_heterogeneous_additive = "Z1=(X1^2-1)/sqrt(2); Z2=sqrt(2)sin(2*pi*X2); eta=(Z1+Z2)/sqrt(2)",
  P4_main_interaction = "Z1=(X1^2-1)/sqrt(2); Z2=X1*X2; eta=(Z1+Z2)/sqrt(2)",
  P5_pure_interaction = "eta=X1*X2"
)

condition_definitions <- data.frame(
  condition = condition_levels,
  condition_id = unname(condition_ids[condition_levels]),
  label = unname(condition_labels[condition_levels]),
  formula_at_rho0 = unname(condition_formulas[condition_levels]),
  primary_interpretation = c(
    "independence baseline",
    "linear benchmark; GP should be near zero",
    "strong additive nonlinear; GP expected large",
    "heterogeneous additive nonlinear; GP expected positive",
    "mixed additive + interaction; intermediate GP/GR",
    "pure interaction; GP relatively small, GR relatively large"
  ),
  stringsAsFactors = FALSE
)

write.csv(condition_definitions,
          file.path(SIM_OUTPUT, "condition_definitions.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# ----------------------------------------------------------------------------
# 2.1. Workload plan shown before the long run
# ----------------------------------------------------------------------------
build_workload_plan <- function() {
  ncond <- length(condition_levels)
  nnonnull <- ncond - 1L

  trackA_tasks <- length(MAIN_N_VALUES) * length(MAIN_P_VALUES) * MAIN_B *
    (1L + nnonnull * length(MAIN_R2_VALUES))

  trackB_datasets <- length(INFER_N_VALUES) * ncond * INFER_REPLICATIONS
  # Each Track-B dataset has one observed cross-fit plus INFER_BOOT_B bootstrap cross-fits.
  trackB_crossfits <- trackB_datasets * (1L + INFER_BOOT_B)

  trackC1_tasks <- length(RHO_VALUES) * ncond * ROBUST_B
  trackC2_tasks <- length(SPAN_VALUES) * ncond * SPAN_B
  trackC3_tasks <- length(ERROR_DISTRIBUTIONS) * ncond * ERROR_B
  trackC4_tasks <- length(KFOLD_VALUES) * ncond * KFOLD_B
  trackD_tasks <- ncond * BENCHMARK_B

  real_cf <- REAL_CF_REPEATS
  real_boot <- 2L * REAL_BOOT_B
  real_span <- length(SPAN_VALUES) * REAL_SENS_REPEATS
  real_kfold <- length(KFOLD_VALUES) * REAL_SENS_REPEATS
  real_bench <- REAL_BENCH_REPEATS
  real_tasks <- real_cf + real_boot + real_span + real_kfold + real_bench

  data.frame(
    track = c(
      "A Main factorial",
      "B Linear-null inference",
      "C1 Correlated X",
      "C2 Span sensitivity",
      "C3 Error robustness",
      "C4 K-fold sensitivity",
      "D Method benchmark",
      "E Concrete real data"
    ),
    top_level_tasks = c(
      trackA_tasks, trackB_datasets, trackC1_tasks, trackC2_tasks,
      trackC3_tasks, trackC4_tasks, trackD_tasks, real_tasks
    ),
    expensive_inner_evaluations = c(
      NA_real_, trackB_crossfits, NA_real_, NA_real_,
      NA_real_, NA_real_, NA_real_, real_boot
    ),
    note = c(
      "one row = one simulated dataset; each includes in-sample + cross-fit analysis",
      paste0("each dataset = 1 observed + ", INFER_BOOT_B, " bootstrap cross-fits"),
      "core Monte Carlo replications",
      "core Monte Carlo replications",
      "core Monte Carlo replications",
      "core Monte Carlo replications",
      "2-fold Linear/ACE/GAM/RF benchmark",
      "repeated CF + 2 bootstraps + sensitivity + benchmark"
    ),
    stringsAsFactors = FALSE
  )
}

print_workload_plan <- function() {
  tab <- build_workload_plan()
  write.csv(tab, file.path(ROOT_OUTPUT, "WORKLOAD_PLAN.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  log_message("============================================================")
  log_message("WORKLOAD PLAN (profile=", RUN_PROFILE, ", workers=", MAX_WORKERS, ")")
  for (i in seq_len(nrow(tab))) {
    inner_txt <- if (is.finite(tab$expensive_inner_evaluations[i])) {
      paste0("; inner evaluations~", format(tab$expensive_inner_evaluations[i], big.mark = ",", scientific = FALSE))
    } else ""
    log_message(
      sprintf("%-24s : %s top-level tasks%s | %s",
              tab$track[i],
              format(tab$top_level_tasks[i], big.mark = ",", scientific = FALSE),
              inner_txt, tab$note[i])
    )
  }
  log_message("ETA is estimated empirically after each completed progress batch; early ETA can fluctuate.")
  log_message("============================================================")
  invisible(tab)
}

# ----------------------------------------------------------------------------
# 3. Generic numerical helpers
# ----------------------------------------------------------------------------
safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x)
}

safe_se <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x) / sqrt(length(x))
}

safe_quantile <- function(x, prob) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else unname(stats::quantile(x, prob = prob, names = FALSE))
}

wilson_interval <- function(x, n, conf = 0.95) {
  if (!is.finite(x) || !is.finite(n) || n <= 0) return(c(NA_real_, NA_real_))
  z <- stats::qnorm(1 - (1 - conf) / 2)
  phat <- x / n
  den <- 1 + z^2 / n
  center <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(max(0, center - half), min(1, center + half))
}

r2_to_noise_sd <- function(target_r2) {
  if (!is.finite(target_r2) || target_r2 <= 0 || target_r2 >= 1) {
    stop("target_r2 must be strictly between 0 and 1 for non-null conditions.")
  }
  sqrt((1 - target_r2) / target_r2)
}

# Biased V-statistic sample distance correlation, matching the pilot code.
distance_correlation <- function(x, y) {
  x <- as.matrix(x)
  y <- as.matrix(y)
  if (nrow(x) != nrow(y)) stop("x and y must have the same number of rows.")
  n <- nrow(x)
  if (n < 2L) stop("At least two observations are required.")

  a <- as.matrix(stats::dist(x))
  b <- as.matrix(stats::dist(y))

  A <- a - rowMeans(a) -
    matrix(colMeans(a), nrow = n, ncol = n, byrow = TRUE) + mean(a)
  B <- b - rowMeans(b) -
    matrix(colMeans(b), nrow = n, ncol = n, byrow = TRUE) + mean(b)

  dcov2_xy <- max(mean(A * B), 0)
  dcov2_xx <- max(mean(A * A), 0)
  dcov2_yy <- max(mean(B * B), 0)

  denominator <- sqrt(sqrt(dcov2_xx * dcov2_yy))
  if (!is.finite(denominator) || denominator <= .Machine$double.eps) return(0)
  sqrt(dcov2_xy) / denominator
}

standardize_train_test <- function(x_train, x_test) {
  mu <- colMeans(x_train)
  s <- apply(x_train, 2, stats::sd)
  s[!is.finite(s) | s <= .Machine$double.eps] <- 1
  train_z <- sweep(sweep(x_train, 2, mu, "-"), 2, s, "/")
  test_z <- sweep(sweep(x_test, 2, mu, "-"), 2, s, "/")
  list(train = train_z, test = test_z, center = mu, scale = s)
}

standardize_full <- function(x) {
  mu <- colMeans(x)
  s <- apply(x, 2, stats::sd)
  s[!is.finite(s) | s <= .Machine$double.eps] <- 1
  z <- sweep(sweep(x, 2, mu, "-"), 2, s, "/")
  list(x = z, center = mu, scale = s)
}

# ----------------------------------------------------------------------------
# 4. Data-generating mechanism
# ----------------------------------------------------------------------------
make_correlated_gaussian <- function(n, p, rho = 0) {
  if (p < 2L) stop("p must be at least 2.")
  if (!is.finite(rho) || abs(rho) >= 1) stop("rho must satisfy |rho| < 1.")

  if (rho == 0) {
    z <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  } else {
    idx <- seq_len(p)
    Sigma <- outer(idx, idx, function(i, j) rho^abs(i - j))
    L <- chol(Sigma)
    z0 <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
    z <- z0 %*% L
  }
  colnames(z) <- paste0("X", seq_len(p))
  z
}

make_x_matrix <- function(n, p, condition, rho = 0) {
  z <- make_correlated_gaussian(n, p, rho)
  if (condition == "P3_heterogeneous_additive") {
    # Gaussian copula dependence while preserving X2 ~ Uniform(0,1).
    z[, 2] <- stats::pnorm(z[, 2])
  }
  colnames(z) <- paste0("X", seq_len(p))
  z
}

# Exact population centering/scaling for the AR(1)-correlated X design.
# At rho=0 these reduce exactly to the pilot P1--P5 definitions with Var(eta)=1.
signal_center_sd <- function(condition, rho) {
  if (condition == "P0_null") return(c(center = 0, sd = 1))
  if (condition == "P1_linear") {
    return(c(center = 0, sd = sqrt(1 + rho)))
  }
  if (condition == "P2_symmetric_additive") {
    return(c(center = 0, sd = sqrt(1 + rho^2)))
  }
  if (condition == "P3_heterogeneous_additive") {
    # Cov((X1^2-1), sin(2*pi*Phi(Z2))) = 0 by symmetry.
    return(c(center = 0, sd = 1))
  }
  if (condition == "P4_main_interaction") {
    center <- rho / sqrt(2)
    variance <- 1 + rho^2 / 2 + sqrt(2) * rho
    return(c(center = center, sd = sqrt(variance)))
  }
  if (condition == "P5_pure_interaction") {
    return(c(center = rho, sd = sqrt(1 + rho^2)))
  }
  stop("Unknown condition: ", condition)
}

raw_signal <- function(x, condition) {
  x1 <- x[, "X1"]
  x2 <- x[, "X2"]

  if (condition == "P0_null") return(rep(0, nrow(x)))
  if (condition == "P1_linear") return((x1 + x2) / sqrt(2))
  if (condition == "P2_symmetric_additive") return((x1^2 + x2^2 - 2) / 2)
  if (condition == "P3_heterogeneous_additive") {
    z1 <- (x1^2 - 1) / sqrt(2)
    z2 <- sqrt(2) * sin(2 * pi * x2)
    return((z1 + z2) / sqrt(2))
  }
  if (condition == "P4_main_interaction") {
    z1 <- (x1^2 - 1) / sqrt(2)
    z2 <- x1 * x2
    return((z1 + z2) / sqrt(2))
  }
  if (condition == "P5_pure_interaction") return(x1 * x2)
  stop("Unknown condition: ", condition)
}

standardized_signal <- function(x, condition, rho = 0) {
  eta0 <- raw_signal(x, condition)
  if (condition == "P0_null") return(eta0)
  cs <- signal_center_sd(condition, rho)
  (eta0 - unname(cs["center"])) / unname(cs["sd"])
}

generate_standard_error <- function(n, error_dist = "normal") {
  if (error_dist == "normal") return(stats::rnorm(n))
  if (error_dist == "t3") return(stats::rt(n, df = 3) / sqrt(3))
  if (error_dist == "centered_exponential") return(stats::rexp(n, rate = 1) - 1)
  stop("Unknown error_dist: ", error_dist)
}

generate_data <- function(condition, n, p, target_r2 = 0.5,
                          rho = 0, error_dist = "normal", seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  x <- make_x_matrix(n, p, condition, rho)
  eta <- standardized_signal(x, condition, rho)
  eps_std <- generate_standard_error(n, error_dist)

  if (condition == "P0_null") {
    noise_sd <- 1
    y <- eps_std
    target_r2_used <- NA_real_
  } else {
    noise_sd <- r2_to_noise_sd(target_r2)
    y <- eta + noise_sd * eps_std
    target_r2_used <- target_r2
  }

  list(x = x, y = as.numeric(y), eta = as.numeric(eta),
       epsilon = noise_sd * eps_std, noise_sd = noise_sd,
       target_r2 = target_r2_used, rho = rho, error_dist = error_dist)
}

# ----------------------------------------------------------------------------
# 5. ACE fitting and out-of-sample transformation
# ----------------------------------------------------------------------------

# Build an acepack control list with types that are accepted by
# acepack::set_control().  In particular, maxit MUST be double/numeric,
# not an integer literal (e.g. 50L).
make_ace_control <- function(span = ACE_SPAN, maxit = ACE_MAXIT) {
  span <- as.double(span)
  maxit <- as.double(maxit)

  if (length(span) != 1L || !is.finite(span) || span < 0 || span > 1) {
    stop("ACE span must be one finite numeric value in [0, 1].")
  }
  if (length(maxit) != 1L || !is.finite(maxit) || maxit <= 0) {
    stop("ACE maxit must be one positive finite numeric value.")
  }
  if (!inherits(span, "numeric") || !inherits(maxit, "numeric")) {
    stop("ACE control values must inherit from class 'numeric'.")
  }

  list(
    span = span,
    maxit = maxit
  )
}

fit_ace_safely <- function(x, y, span = ACE_SPAN) {
  tryCatch({
    t0 <- proc.time()[3]
    fit <- acepack::ace(
      x = x,
      y = y,
      delrsq = 1e-4,
      control = make_ace_control(span = span, maxit = ACE_MAXIT),
      on.error = function(message) stop(message)
    )
    elapsed <- proc.time()[3] - t0
    tx <- as.matrix(fit$tx)
    ty <- as.numeric(fit$ty)
    score <- rowSums(tx)
    if (any(!is.finite(score)) || any(!is.finite(ty))) {
      stop("ACE returned non-finite transformed values.")
    }
    list(success = TRUE, fit = fit, tx = tx, ty = ty, score = score,
         elapsed_seconds = as.numeric(elapsed), error_message = NA_character_)
  }, error = function(e) {
    list(success = FALSE, fit = NULL, tx = NULL, ty = NULL, score = NULL,
         elapsed_seconds = NA_real_, error_message = conditionMessage(e))
  })
}

make_interp_function <- function(x_train, z_train) {
  ok <- is.finite(x_train) & is.finite(z_train)
  x_train <- x_train[ok]
  z_train <- z_train[ok]

  if (length(x_train) == 0L) return(function(x_new) rep(0, length(x_new)))
  if (length(x_train) < 4L || !is.finite(stats::sd(z_train)) ||
      stats::sd(z_train) <= .Machine$double.eps) {
    z0 <- mean(z_train)
    return(function(x_new) rep(z0, length(x_new)))
  }

  z_by_x <- tapply(z_train, x_train, mean)
  xu <- as.numeric(names(z_by_x))
  zu <- as.numeric(z_by_x)
  ord <- order(xu)
  xu <- xu[ord]
  zu <- zu[ord]
  if (length(xu) == 1L) {
    z0 <- zu[1L]
    return(function(x_new) rep(z0, length(x_new)))
  }
  f <- stats::approxfun(xu, zu, rule = 2)
  function(x_new) as.numeric(f(x_new))
}

apply_fitted_ace_to_test <- function(ace_result, x_train, y_train, x_test, y_test) {
  if (!ace_result$success) {
    return(list(success = FALSE, R2 = NA_real_, R3 = NA_real_,
                score_test = NULL, ty_test = NULL,
                error_message = ace_result$error_message))
  }

  p <- ncol(x_train)
  tx_test <- matrix(NA_real_, nrow = nrow(x_test), ncol = p)
  for (j in seq_len(p)) {
    fj <- make_interp_function(x_train[, j], ace_result$tx[, j])
    tx_test[, j] <- fj(x_test[, j])
  }
  theta_fun <- make_interp_function(y_train, ace_result$ty)
  ty_test <- theta_fun(y_test)
  score_test <- rowSums(tx_test)

  if (any(!is.finite(score_test)) || any(!is.finite(ty_test))) {
    return(list(success = FALSE, R2 = NA_real_, R3 = NA_real_,
                score_test = NULL, ty_test = NULL,
                error_message = "Non-finite test transformation."))
  }

  list(success = TRUE,
       R2 = distance_correlation(score_test, y_test),
       R3 = distance_correlation(score_test, ty_test),
       score_test = score_test, ty_test = ty_test,
       error_message = NA_character_)
}

make_fold_id <- function(n, k = CF_FOLDS, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  fold_id <- rep(seq_len(k), length.out = n)
  sample(fold_id, size = n, replace = FALSE)
}

# ----------------------------------------------------------------------------
# 5.1. ACE startup self-test
# ----------------------------------------------------------------------------
# This is intentionally run before any long Monte Carlo work.  The previous
# failed run showed that a silent ACE failure can otherwise waste hours while
# still producing R0/R1.  A successful self-test must return finite tx/ty/rsq.
ace_startup_selftest <- function() {
  # First validate the exact control object that every ACE fit will use.
  ctrl <- make_ace_control(ACE_SPAN, ACE_MAXIT)

  if (!identical(typeof(ctrl$span), "double") ||
      !identical(typeof(ctrl$maxit), "double")) {
    stop(
      paste0(
        "ACE startup self-test FAILED before model fitting: ",
        "control types are span=", typeof(ctrl$span),
        ", maxit=", typeof(ctrl$maxit),
        ". Both must be double."
      )
    )
  }

  # Test acepack::set_control directly so a package-interface mismatch is
  # caught before tryCatch inside fit_ace_safely can hide it.
  ctrl_test <- tryCatch(
    {
      do.call(acepack::set_control, ctrl)
      TRUE
    },
    error = function(e) e
  )

  if (!isTRUE(ctrl_test)) {
    stop(
      paste0(
        "ACE startup self-test FAILED in acepack::set_control(). ",
        "acepack version=", ACEPACK_VERSION,
        "; span type=", typeof(ctrl$span),
        "; maxit type=", typeof(ctrl$maxit),
        "; error: ", conditionMessage(ctrl_test),
        "\nNo long simulation has been started."
      )
    )
  }

  set.seed(BASE_SEED + 4242L)
  n_test <- 120L
  x_test <- cbind(
    X1 = stats::rnorm(n_test),
    X2 = stats::rnorm(n_test)
  )
  y_test <- x_test[, 1]^2 + 0.5 * x_test[, 2] + stats::rnorm(n_test, sd = 0.5)

  ans <- fit_ace_safely(x_test, y_test, span = ACE_SPAN)
  if (!isTRUE(ans$success)) {
    stop(
      paste0(
        "ACE startup self-test FAILED. acepack version=", ACEPACK_VERSION,
        "; error: ", ans$error_message,
        "\nNo long simulation has been started."
      )
    )
  }
  if (nrow(ans$tx) != n_test || ncol(ans$tx) != 2L ||
      !all(is.finite(ans$tx)) || !all(is.finite(ans$ty)) ||
      !is.finite(ans$fit$rsq)) {
    stop("ACE startup self-test returned malformed/non-finite output.")
  }

  log_message(
    "[ACE SELF-TEST PASS] acepack=", ACEPACK_VERSION,
    "; span=", ACE_SPAN,
    "; maxit=", ACE_MAXIT,
    "; control types=(span:", typeof(ctrl$span),
    ", maxit:", typeof(ctrl$maxit), ")",
    "; rsq=", sprintf("%.4f", ans$fit$rsq),
    "; elapsed=", sprintf("%.3fs", ans$elapsed_seconds)
  )
  invisible(TRUE)
}

# ----------------------------------------------------------------------------
# 6. Stagewise metrics: in-sample and cross-fitted
# ----------------------------------------------------------------------------
insample_metrics <- function(x, y, span = ACE_SPAN, standardize_x = FALSE) {
  x_raw <- as.matrix(x)
  if (standardize_x) {
    sx <- standardize_full(x_raw)
    x_use <- sx$x
  } else {
    x_use <- x_raw
  }

  R0_raw <- distance_correlation(x_raw, y)
  R0 <- if (standardize_x) distance_correlation(x_use, y) else R0_raw

  lin_df <- data.frame(y = y, x_use, check.names = FALSE)
  lin_fit <- stats::lm(y ~ ., data = lin_df)
  linear_score <- as.numeric(stats::fitted(lin_fit))
  R1 <- distance_correlation(linear_score, y)

  ace_fit <- fit_ace_safely(x_use, y, span)
  if (ace_fit$success) {
    R2 <- distance_correlation(ace_fit$score, y)
    R3 <- distance_correlation(ace_fit$score, ace_fit$ty)
    ace_rsq <- unname(ace_fit$fit$rsq)
    tx_var <- apply(ace_fit$tx, 2, stats::var)
    total_tx_var <- sum(tx_var, na.rm = TRUE)
    if (length(tx_var) >= 3L && is.finite(total_tx_var) && total_tx_var > 0) {
      noise_contribution_ratio <- sum(tx_var[3:length(tx_var)], na.rm = TRUE) / total_tx_var
    } else {
      noise_contribution_ratio <- 0
    }
  } else {
    R2 <- R3 <- ace_rsq <- noise_contribution_ratio <- NA_real_
  }

  list(
    R0 = R0, R0_raw = R0_raw, R1 = R1, R2 = R2, R3 = R3,
    GP = R2 - R1, GR = R3 - R2,
    delta10 = R1 - R0, delta20 = R2 - R0, delta30 = R3 - R0,
    ace_rsq = ace_rsq, noise_contribution_ratio = noise_contribution_ratio,
    ace_success = ace_fit$success,
    ace_error_message = ace_fit$error_message,
    ace_fit = ace_fit, linear_fit = lin_fit
  )
}

crossfit_metrics <- function(x, y, k = CF_FOLDS, span = ACE_SPAN,
                             seed = NULL, fold_id = NULL,
                             standardize_x = FALSE) {
  x <- as.matrix(x)
  if (is.null(fold_id)) fold_id <- make_fold_id(nrow(x), k, seed)
  folds <- sort(unique(fold_id))
  rows <- vector("list", length(folds))

  for (ii in seq_along(folds)) {
    fold <- folds[ii]
    test_index <- which(fold_id == fold)
    train_index <- which(fold_id != fold)

    x_train_raw <- x[train_index, , drop = FALSE]
    x_test_raw <- x[test_index, , drop = FALSE]
    y_train <- y[train_index]
    y_test <- y[test_index]

    if (standardize_x) {
      sc <- standardize_train_test(x_train_raw, x_test_raw)
      x_train <- sc$train
      x_test <- sc$test
    } else {
      x_train <- x_train_raw
      x_test <- x_test_raw
    }

    R0_raw <- distance_correlation(x_test_raw, y_test)
    R0 <- if (standardize_x) distance_correlation(x_test, y_test) else R0_raw

    train_df <- data.frame(y = y_train, x_train, check.names = FALSE)
    test_df <- data.frame(x_test, check.names = FALSE)
    lin_fit <- stats::lm(y ~ ., data = train_df)
    lin_pred <- suppressWarnings(as.numeric(stats::predict(lin_fit, newdata = test_df)))
    R1 <- if (all(is.finite(lin_pred))) distance_correlation(lin_pred, y_test) else NA_real_

    ace_fit <- fit_ace_safely(x_train, y_train, span)
    ace_test <- apply_fitted_ace_to_test(ace_fit, x_train, y_train, x_test, y_test)
    R2 <- ace_test$R2
    R3 <- ace_test$R3

    rows[[ii]] <- data.frame(
      fold = fold,
      n_train = length(train_index),
      n_test = length(test_index),
      R0 = R0, R0_raw = R0_raw, R1 = R1, R2 = R2, R3 = R3,
      GP = R2 - R1, GR = R3 - R2,
      delta10 = R1 - R0, delta20 = R2 - R0, delta30 = R3 - R0,
      ace_success = ace_test$success,
      ace_error_message = if (is.null(ace_test$error_message)) NA_character_ else ace_test$error_message,
      stringsAsFactors = FALSE
    )
  }

  tab <- do.call(rbind, rows)
  list(
    success = all(tab$ace_success),
    success_rate = mean(tab$ace_success),
    R0 = safe_mean(tab$R0), R0_raw = safe_mean(tab$R0_raw),
    R1 = safe_mean(tab$R1), R2 = safe_mean(tab$R2), R3 = safe_mean(tab$R3),
    GP = safe_mean(tab$GP), GR = safe_mean(tab$GR),
    delta10 = safe_mean(tab$delta10), delta20 = safe_mean(tab$delta20),
    delta30 = safe_mean(tab$delta30),
    error_message = {
      ee <- unique(tab$ace_error_message[!is.na(tab$ace_error_message) & nzchar(tab$ace_error_message)])
      if (length(ee) == 0L) NA_character_ else paste(ee, collapse = " | ")
    },
    fold_table = tab, fold_id = fold_id
  )
}

# ----------------------------------------------------------------------------
# 7. Single Monte Carlo replication
# ----------------------------------------------------------------------------
replication_seed <- function(condition, n, p, replication, target_r2 = 0.5,
                             seed_block = 0L) {
  ci <- match(condition, condition_levels)
  r2code <- if (is.finite(target_r2)) as.integer(round(target_r2 * 1000)) else 0L
  as.integer(BASE_SEED + seed_block + ci * 10000000L +
               n * 10000L + p * 100L + r2code * 3L + replication)
}

run_core_replication <- function(condition, n, p, replication,
                                 target_r2 = 0.5, rho = 0,
                                 error_dist = "normal", span = ACE_SPAN,
                                 k = CF_FOLDS, standardize_x = SIM_STANDARDIZE_X,
                                 seed_block = 0L) {
  seed <- replication_seed(condition, n, p, replication, target_r2, seed_block)
  t_total <- proc.time()[3]

  dat <- generate_data(condition, n, p, target_r2, rho, error_dist, seed)
  ins <- insample_metrics(dat$x, dat$y, span, standardize_x)
  cf <- crossfit_metrics(dat$x, dat$y, k, span,
                         seed = seed + 500000L,
                         standardize_x = standardize_x)

  elapsed <- proc.time()[3] - t_total

  data.frame(
    condition = condition,
    condition_id = unname(condition_ids[condition]),
    n = n, p = p,
    target_r2 = if (condition == "P0_null") NA_real_ else target_r2,
    rho = rho,
    error_dist = error_dist,
    span = span,
    k = k,
    replication = replication,
    seed = seed,
    R0 = ins$R0, R0_raw = ins$R0_raw, R1 = ins$R1, R2 = ins$R2, R3 = ins$R3,
    GP = ins$GP, GR = ins$GR,
    delta10 = ins$delta10, delta20 = ins$delta20, delta30 = ins$delta30,
    cf_R0 = cf$R0, cf_R0_raw = cf$R0_raw,
    cf_R1 = cf$R1, cf_R2 = cf$R2, cf_R3 = cf$R3,
    cf_GP = cf$GP, cf_GR = cf$GR,
    cf_delta10 = cf$delta10, cf_delta20 = cf$delta20, cf_delta30 = cf$delta30,
    ace_rsq = ins$ace_rsq,
    noise_contribution_ratio = ins$noise_contribution_ratio,
    ace_success = ins$ace_success,
    cf_success = cf$success,
    cf_success_rate = cf$success_rate,
    ace_error_message = if (is.null(ins$ace_error_message)) NA_character_ else ins$ace_error_message,
    cf_error_message = if (is.null(cf$error_message)) NA_character_ else cf$error_message,
    runtime_seconds = as.numeric(elapsed),
    stringsAsFactors = FALSE
  )
}

# ----------------------------------------------------------------------------
# 7.1. End-to-end startup self-test
# ----------------------------------------------------------------------------
# This tests the same path used in Track A:
# data generation -> in-sample ACE -> 2-fold cross-fitting -> test interpolation
# -> R0/R1/R2/R3/GP/GR.  It runs only once and is intentionally small.
pipeline_startup_selftest <- function() {
  test_rep <- tryCatch(
    run_core_replication(
      condition = "P2_symmetric_additive",
      n = 200L,
      p = 5L,
      replication = 1L,
      target_r2 = 0.5,
      rho = 0,
      error_dist = "normal",
      span = ACE_SPAN,
      k = CF_FOLDS,
      standardize_x = SIM_STANDARDIZE_X,
      seed_block = 910000000L
    ),
    error = function(e) e
  )

  if (inherits(test_rep, "error")) {
    stop(
      paste0(
        "PIPELINE startup self-test FAILED with an R error: ",
        conditionMessage(test_rep),
        "\nNo long simulation has been started."
      )
    )
  }

  required_numeric <- c(
    "R0", "R1", "R2", "R3", "GP", "GR",
    "cf_R0", "cf_R1", "cf_R2", "cf_R3", "cf_GP", "cf_GR"
  )

  missing_cols <- setdiff(required_numeric, names(test_rep))
  if (length(missing_cols) > 0L) {
    stop(
      paste0(
        "PIPELINE startup self-test FAILED: missing output columns: ",
        paste(missing_cols, collapse = ", "),
        "\nNo long simulation has been started."
      )
    )
  }

  vals <- unlist(test_rep[1, required_numeric, drop = FALSE], use.names = TRUE)
  good_numeric <- all(is.finite(as.numeric(vals)))
  good_flags <- isTRUE(test_rep$ace_success[1]) && isTRUE(test_rep$cf_success[1])

  if (!good_numeric || !good_flags) {
    stop(
      paste0(
        "PIPELINE startup self-test FAILED. ",
        "ace_success=", test_rep$ace_success[1],
        "; cf_success=", test_rep$cf_success[1],
        "; ace_error=", ifelse(is.na(test_rep$ace_error_message[1]), "NA", test_rep$ace_error_message[1]),
        "; cf_error=", ifelse(is.na(test_rep$cf_error_message[1]), "NA", test_rep$cf_error_message[1]),
        "; finite metrics=", good_numeric,
        "\nNo long simulation has been started."
      )
    )
  }

  log_message(
    "[PIPELINE SELF-TEST PASS] ",
    "R2=", sprintf("%.4f", test_rep$R2[1]),
    "; R3=", sprintf("%.4f", test_rep$R3[1]),
    "; GP=", sprintf("%.4f", test_rep$GP[1]),
    "; GR=", sprintf("%.4f", test_rep$GR[1]),
    "; cf_R2=", sprintf("%.4f", test_rep$cf_R2[1]),
    "; cf_R3=", sprintf("%.4f", test_rep$cf_R3[1]),
    "; cf_GP=", sprintf("%.4f", test_rep$cf_GP[1]),
    "; cf_GR=", sprintf("%.4f", test_rep$cf_GR[1])
  )

  invisible(TRUE)
}


# ----------------------------------------------------------------------------
# 8. Summary functions
# ----------------------------------------------------------------------------
summarize_mc <- function(results, group_vars) {
  metrics <- c(
    "R0", "R1", "R2", "R3", "GP", "GR", "delta10", "delta20", "delta30",
    "cf_R0", "cf_R1", "cf_R2", "cf_R3", "cf_GP", "cf_GR",
    "cf_delta10", "cf_delta20", "cf_delta30", "ace_rsq",
    "noise_contribution_ratio", "runtime_seconds"
  )

  keys <- unique(results[, group_vars, drop = FALSE])
  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    key <- keys[i, , drop = FALSE]
    keep <- rep(TRUE, nrow(results))
    for (v in group_vars) {
      kv <- key[[v]][1]
      if (is.na(kv)) keep <- keep & is.na(results[[v]])
      else keep <- keep & (results[[v]] == kv)
    }
    sub <- results[keep, , drop = FALSE]
    row <- key
    row$replications <- nrow(sub)
    row$ace_success_rate <- mean(sub$ace_success, na.rm = TRUE)
    row$cf_success_rate <- mean(sub$cf_success, na.rm = TRUE)

    for (m in metrics) {
      mn <- safe_mean(sub[[m]])
      se <- safe_se(sub[[m]])
      row[[paste0(m, "_mean")]] <- mn
      row[[paste0(m, "_sd")]] <- safe_sd(sub[[m]])
      row[[paste0(m, "_se")]] <- se
      row[[paste0(m, "_mc95_lower")]] <- if (is.finite(mn) && is.finite(se)) mn - 1.96 * se else NA_real_
      row[[paste0(m, "_mc95_upper")]] <- if (is.finite(mn) && is.finite(se)) mn + 1.96 * se else NA_real_
    }
    rows[[i]] <- row
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}


# Safe grouped means that never fail when a response column is all NA.
# Unlike aggregate(..., formula=...), this preserves groups and returns NA
# for a metric when that group contains no finite values.
safe_group_means <- function(data, group_vars, value_vars) {
  if (nrow(data) == 0L) {
    out <- data.frame(matrix(nrow = 0L, ncol = length(group_vars) + length(value_vars)))
    names(out) <- c(group_vars, value_vars)
    return(out)
  }

  keys <- unique(data[, group_vars, drop = FALSE])
  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    key <- keys[i, , drop = FALSE]
    keep <- rep(TRUE, nrow(data))
    for (v in group_vars) {
      kv <- key[[v]][1]
      if (is.na(kv)) keep <- keep & is.na(data[[v]])
      else keep <- keep & (data[[v]] == kv)
    }
    sub <- data[keep, , drop = FALSE]
    row <- key
    for (m in value_vars) row[[m]] <- safe_mean(sub[[m]])
    rows[[i]] <- row
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

add_p0_adjustment <- function(summary_table) {
  for (m in c("cf_GP_mean", "cf_GR_mean", "cf_delta20_mean", "cf_delta30_mean")) {
    summary_table[[paste0(m, "_p0adj")]] <- NA_real_
  }

  for (i in seq_len(nrow(summary_table))) {
    if (summary_table$condition[i] == "P0_null") {
      for (m in c("cf_GP_mean", "cf_GR_mean", "cf_delta20_mean", "cf_delta30_mean")) {
        summary_table[[paste0(m, "_p0adj")]][i] <- 0
      }
      next
    }
    idx0 <- summary_table$condition == "P0_null" &
      summary_table$n == summary_table$n[i] &
      summary_table$p == summary_table$p[i]
    if (sum(idx0) < 1L) next
    j <- which(idx0)[1]
    for (m in c("cf_GP_mean", "cf_GR_mean", "cf_delta20_mean", "cf_delta30_mean")) {
      summary_table[[paste0(m, "_p0adj")]][i] <- summary_table[[m]][i] - summary_table[[m]][j]
    }
  }
  summary_table
}

# ----------------------------------------------------------------------------
# 9. Parallel execution helper with checkpointing
# ----------------------------------------------------------------------------
format_duration <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("NA")
  seconds <- round(seconds)
  d <- seconds %/% 86400L
  h <- (seconds %% 86400L) %/% 3600L
  m <- (seconds %% 3600L) %/% 60L
  sec <- seconds %% 60L
  if (d > 0L) return(sprintf("%dd %02dh %02dm", d, h, m))
  if (h > 0L) return(sprintf("%02dh %02dm %02ds", h, m, sec))
  if (m > 0L) return(sprintf("%02dm %02ds", m, sec))
  sprintf("%02ds", sec)
}

format_eta_clock <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("NA")
  format(Sys.time() + seconds, "%Y-%m-%d %H:%M:%S")
}

parallel_rows <- function(grid, fun, label = "tasks",
                          batch_size = PROGRESS_BATCH_STANDARD,
                          overall_completed_before = 0L,
                          overall_total = NULL,
                          overall_label = NULL) {
  if (nrow(grid) == 0L) return(data.frame())

  total <- nrow(grid)
  batch_size <- max(1L, min(as.integer(batch_size), total))
  ids <- seq_len(total)
  batch_id <- ceiling(ids / batch_size)
  batches <- split(ids, batch_id)
  parts <- vector("list", length(batches))

  t0 <- Sys.time()
  completed <- 0L
  log_message("Starting ", label, ": ", total,
              " tasks; workers=", MAX_WORKERS,
              "; progress batch=", batch_size)

  for (bi in seq_along(batches)) {
    idx <- batches[[bi]]
    g <- grid[idx, , drop = FALSE]
    row_list <- split(g, seq_len(nrow(g)))

    part <- future.apply::future_lapply(
      row_list,
      fun,
      future.seed = TRUE,
      future.scheduling = 2
    )
    parts[[bi]] <- do.call(rbind, part)
    completed <- completed + length(idx)

    if (SHOW_PROGRESS) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      pct <- 100 * completed / total
      rate_per_min <- if (elapsed > 0) 60 * completed / elapsed else NA_real_
      eta_sec <- if (completed > 0 && elapsed > 0)
        elapsed * (total - completed) / completed else NA_real_

      local_txt <- sprintf(
        "[%s] %d/%d (%.1f%%) | elapsed %s | ETA %s (finish ~ %s) | %.2f tasks/min",
        label, completed, total, pct, format_duration(elapsed),
        format_duration(eta_sec), format_eta_clock(eta_sec), rate_per_min
      )

      if (!is.null(overall_total) && is.finite(overall_total) && overall_total > 0) {
        overall_done <- min(overall_total, overall_completed_before + completed)
        overall_pct <- 100 * overall_done / overall_total
        scope <- if (is.null(overall_label)) "overall" else overall_label
        local_txt <- paste0(
          local_txt,
          sprintf(" | %s %d/%d (%.1f%% tasks)",
                  scope, overall_done, overall_total, overall_pct)
        )
      }
      log_message(local_txt)
    }
  }

  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  elapsed_total <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  log_message("Completed ", label, ": ", nrow(out),
              " rows; elapsed=", format_duration(elapsed_total))
  out
}

# ----------------------------------------------------------------------------
# 10. TRACK A: Main factorial simulation
# ----------------------------------------------------------------------------
run_main_factorial <- function() {
  log_message("[Track A] Main factorial simulation started.")

  nonnull_grid <- expand.grid(
    n = MAIN_N_VALUES,
    p = MAIN_P_VALUES,
    target_r2 = MAIN_R2_VALUES,
    condition = setdiff(condition_levels, "P0_null"),
    replication = seq_len(MAIN_B),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  p0_grid <- expand.grid(
    n = MAIN_N_VALUES,
    p = MAIN_P_VALUES,
    condition = "P0_null",
    replication = seq_len(MAIN_B),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  p0_grid$target_r2 <- NA_real_
  p0_grid <- p0_grid[, names(nonnull_grid)]
  full_grid <- rbind(p0_grid, nonnull_grid)

  all_blocks <- list()
  block_ix <- 1L
  track_total_tasks <- nrow(full_grid)
  track_completed_tasks <- 0L
  total_blocks <- length(MAIN_N_VALUES) * length(MAIN_P_VALUES)
  current_block <- 0L
  for (n0 in MAIN_N_VALUES) {
    for (p0 in MAIN_P_VALUES) {
      current_block <- current_block + 1L
      checkpoint <- file.path(
        CHECKPOINT_OUTPUT,
        paste0("trackA_main_n", n0, "_p", p0, ".rds")
      )
      expected <- sum(full_grid$n == n0 & full_grid$p == p0)

      if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
        tmp <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
        if (!is.null(tmp) && nrow(tmp) == expected) {
          track_completed_tasks <- track_completed_tasks + expected
          log_message("[Track A] Reusing checkpoint for n=", n0, ", p=", p0,
                      " | block ", current_block, "/", total_blocks,
                      " | Track A overall ", track_completed_tasks, "/",
                      track_total_tasks, " (",
                      sprintf("%.1f", 100 * track_completed_tasks / track_total_tasks), "% tasks)")
          all_blocks[[block_ix]] <- tmp
          block_ix <- block_ix + 1L
          next
        }
      }

      g <- full_grid[full_grid$n == n0 & full_grid$p == p0, , drop = FALSE]
      res <- parallel_rows(g, function(z) {
        cond <- z$condition[1]
        r2 <- if (cond == "P0_null") 0.5 else z$target_r2[1]
        run_core_replication(
          condition = cond,
          n = as.integer(z$n[1]),
          p = as.integer(z$p[1]),
          replication = as.integer(z$replication[1]),
          target_r2 = as.numeric(r2),
          rho = 0,
          error_dist = "normal",
          span = ACE_SPAN,
          k = CF_FOLDS,
          standardize_x = SIM_STANDARDIZE_X,
          seed_block = 0L
        )
      }, paste0("Track A block ", current_block, "/", total_blocks,
                     " n=", n0, ", p=", p0),
      batch_size = PROGRESS_BATCH_MAIN,
      overall_completed_before = track_completed_tasks,
      overall_total = track_total_tasks,
      overall_label = "Track A overall")

      track_completed_tasks <- track_completed_tasks + nrow(res)
      saveRDS(res, checkpoint)
      all_blocks[[block_ix]] <- res
      block_ix <- block_ix + 1L
    }
  }

  results <- do.call(rbind, all_blocks)
  rownames(results) <- NULL

  # SAVE FIRST: never risk losing a completed long Track A run because a later
  # summarization/plotting step fails. The 12 block checkpoints are also retained.
  trackA_raw_csv <- file.path(SIM_OUTPUT, "A_main_factorial_replications.csv")
  trackA_raw_rds <- file.path(SIM_OUTPUT, "A_main_factorial_replications.rds")
  write.csv(results, trackA_raw_csv, row.names = FALSE, fileEncoding = "UTF-8")
  saveRDS(results, trackA_raw_rds)
  log_message("[Track A] Raw combined results saved: ",
              normalizePath(trackA_raw_csv, winslash = "/", mustWork = FALSE))

  # Basic integrity diagnostics are saved before any higher-level aggregation.
  integrity <- data.frame(
    n_rows = nrow(results),
    expected_rows = track_total_tasks,
    ace_success_rate = safe_mean(as.numeric(results$ace_success)),
    cf_success_rate = safe_mean(as.numeric(results$cf_success)),
    finite_cf_GP_rate = mean(is.finite(results$cf_GP)),
    finite_cf_GR_rate = mean(is.finite(results$cf_GR)),
    stringsAsFactors = FALSE
  )
  write.csv(integrity, file.path(SIM_OUTPUT, "A_trackA_integrity_check.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # If any ACE fits failed, save compact error-frequency tables for diagnosis.
  if ("ace_error_message" %in% names(results)) {
    ee <- results$ace_error_message[!is.na(results$ace_error_message) & nzchar(results$ace_error_message)]
    if (length(ee) > 0L) {
      tab_ee <- as.data.frame(sort(table(ee), decreasing = TRUE), stringsAsFactors = FALSE)
      names(tab_ee) <- c("error_message", "count")
      write.csv(tab_ee, file.path(SIM_OUTPUT, "A_ACE_insample_error_frequencies.csv"),
                row.names = FALSE, fileEncoding = "UTF-8")
    }
  }
  if ("cf_error_message" %in% names(results)) {
    ee <- results$cf_error_message[!is.na(results$cf_error_message) & nzchar(results$cf_error_message)]
    if (length(ee) > 0L) {
      tab_ee <- as.data.frame(sort(table(ee), decreasing = TRUE), stringsAsFactors = FALSE)
      names(tab_ee) <- c("error_message", "count")
      write.csv(tab_ee, file.path(SIM_OUTPUT, "A_ACE_crossfit_error_frequencies.csv"),
                row.names = FALSE, fileEncoding = "UTF-8")
    }
  }

  log_message("[Track A] Integrity: rows=", integrity$n_rows,
              "/", integrity$expected_rows,
              "; ACE success=", sprintf("%.3f", integrity$ace_success_rate),
              "; CF success=", sprintf("%.3f", integrity$cf_success_rate),
              "; finite GP=", sprintf("%.3f", integrity$finite_cf_GP_rate),
              "; finite GR=", sprintf("%.3f", integrity$finite_cf_GR_rate))

  # Publication run safety gate. If ACE/cross-fit failed systematically, do not
  # spend days on Track B--D. The raw results and all checkpoints are already saved.
  if (!is.finite(integrity$finite_cf_GP_rate) || !is.finite(integrity$finite_cf_GR_rate) ||
      integrity$finite_cf_GP_rate < 0.95 || integrity$finite_cf_GR_rate < 0.95) {
    fail_msg <- paste0(
      "Track A integrity check failed: finite cf_GP rate=",
      sprintf("%.3f", integrity$finite_cf_GP_rate),
      ", finite cf_GR rate=", sprintf("%.3f", integrity$finite_cf_GR_rate),
      ". Raw results/checkpoints were preserved. Inspect A_trackA_integrity_check.csv before continuing."
    )
    writeLines(fail_msg, file.path(SIM_OUTPUT, "A_TRACKA_INTEGRITY_STOP.txt"))
    stop(fail_msg)
  }

  summary <- summarize_mc(results, c("condition", "condition_id", "n", "p", "target_r2"))
  summary <- add_p0_adjustment(summary)
  write.csv(summary, file.path(SIM_OUTPUT, "A_main_factorial_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Overall map by signal strength, averaged over n and p.
  # Do NOT use aggregate(formula=...) here: aggregate silently removes rows
  # containing NA in any response and throws 'no rows to aggregate' if all are removed.
  nonnull <- summary[!is.na(summary$condition) & summary$condition != "P0_null", , drop = FALSE]
  overall <- safe_group_means(
    nonnull,
    group_vars = c("condition", "condition_id", "target_r2"),
    value_vars = c("cf_GP_mean", "cf_GR_mean")
  )
  write.csv(overall, file.path(SIM_OUTPUT, "A_GP_GR_overall_by_R2.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Plot only when finite values exist. Missing ACE metrics should create a
  # diagnostic warning/file, not crash the entire multi-day experiment.
  finite_overall <- overall[is.finite(overall$cf_GP_mean) & is.finite(overall$cf_GR_mean), , drop = FALSE]
  if (nrow(finite_overall) > 0L) {
    pdf(file.path(SIM_OUTPUT, "A_GP_GR_diagnostic_maps.pdf"), width = 8, height = 7, onefile = TRUE)
    for (r2 in MAIN_R2_VALUES) {
      d <- finite_overall[is.finite(finite_overall$target_r2) & finite_overall$target_r2 == r2, , drop = FALSE]
      if (nrow(d) == 0L) {
        plot.new()
        title(main = paste0("Diagnostic map, target R^2 = ", r2))
        text(0.5, 0.5, "No finite GP/GR values available")
        next
      }
      xr <- range(c(0, d$cf_GP_mean), finite = TRUE)
      yr <- range(c(0, d$cf_GR_mean), finite = TRUE)
      if (diff(xr) == 0) xr <- xr + c(-0.01, 0.01)
      if (diff(yr) == 0) yr <- yr + c(-0.01, 0.01)
      plot(d$cf_GP_mean, d$cf_GR_mean, pch = 19,
           xlim = xr, ylim = yr,
           xlab = "Cross-fitted GP = R2 - R1",
           ylab = "Cross-fitted GR = R3 - R2",
           main = paste0("Diagnostic map, target R^2 = ", r2))
      abline(h = 0, v = 0, lty = 2)
      text(d$cf_GP_mean, d$cf_GR_mean, labels = d$condition_id, pos = 3)
    }
    dev.off()
  } else {
    warning_msg <- paste0(
      "Track A completed, but no finite cross-fitted GP/GR values were available. ",
      "Inspect A_trackA_integrity_check.csv and A_main_factorial_replications.csv."
    )
    log_message("[Track A WARNING] ", warning_msg)
    writeLines(warning_msg, file.path(SIM_OUTPUT, "A_GP_GR_NO_FINITE_VALUES_WARNING.txt"))
  }

  pdf(file.path(SIM_OUTPUT, "A_GP_GR_by_n_p.pdf"), width = 10, height = 7, onefile = TRUE)
  for (r2 in MAIN_R2_VALUES) {
    for (metric in c("cf_GP_mean", "cf_GR_mean")) {
      d0 <- summary[
        !is.na(summary$condition) & summary$condition != "P0_null" &
          is.finite(summary$target_r2) & summary$target_r2 == r2,
        , drop = FALSE
      ]
      vals <- d0[[metric]]
      vals <- vals[is.finite(vals)]
      yr <- if (length(vals) > 0L) range(c(0, vals), finite = TRUE) else c(-0.01, 0.01)
      if (diff(yr) == 0) yr <- yr + c(-0.01, 0.01)
      for (n0 in MAIN_N_VALUES) {
        plot(NA, xlim = range(MAIN_P_VALUES), ylim = yr,
             xlab = "Number of predictors p",
             ylab = metric,
             main = paste0(metric, ", target R^2=", r2, ", n=", n0))
        for (cond in setdiff(condition_levels, "P0_null")) {
          d <- d0[d0$n == n0 & d0$condition == cond, , drop = FALSE]
          d <- d[order(d$p), , drop = FALSE]
          ok <- is.finite(d[[metric]])
          if (any(ok)) {
            lines(d$p[ok], d[[metric]][ok], type = "b",
                  pch = match(cond, condition_levels),
                  lty = match(cond, condition_levels))
          }
        }
        legend("topright", legend = condition_ids[setdiff(condition_levels, "P0_null")],
               pch = match(setdiff(condition_levels, "P0_null"), condition_levels),
               lty = match(setdiff(condition_levels, "P0_null"), condition_levels),
               bty = "n", cex = 0.8)
      }
    }
  }
  dev.off()

  log_message("[Track A] Main factorial simulation completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 11. TRACK B: Linear-null residual-bootstrap inference
# ----------------------------------------------------------------------------
run_linear_null_bootstrap_rep <- function(condition, n, p, target_r2,
                                          replication, n_boot = INFER_BOOT_B,
                                          span = ACE_SPAN) {
  seed <- replication_seed(condition, n, p, replication, target_r2, 200000000L)
  dat <- generate_data(condition, n, p, target_r2, rho = 0,
                       error_dist = "normal", seed = seed)

  fold_id <- make_fold_id(n, CF_FOLDS, seed + 97L)
  observed <- crossfit_metrics(dat$x, dat$y, CF_FOLDS, span,
                               fold_id = fold_id,
                               standardize_x = SIM_STANDARDIZE_X)

  lin_df <- data.frame(y = dat$y, dat$x, check.names = FALSE)
  lin_fit <- stats::lm(y ~ ., data = lin_df)
  fitted0 <- as.numeric(stats::fitted(lin_fit))
  residual0 <- as.numeric(stats::residuals(lin_fit))
  residual0 <- residual0 - mean(residual0)

  boot_gp <- rep(NA_real_, n_boot)
  boot_gr <- rep(NA_real_, n_boot)
  for (b in seq_len(n_boot)) {
    set.seed(seed + 100000L + b)
    e_boot <- sample(residual0, size = n, replace = TRUE)
    y_boot <- fitted0 + e_boot
    tmp <- crossfit_metrics(dat$x, y_boot, CF_FOLDS, span,
                            fold_id = fold_id,
                            standardize_x = SIM_STANDARDIZE_X)
    boot_gp[b] <- tmp$GP
    boot_gr[b] <- tmp$GR
  }

  valid_gp <- boot_gp[is.finite(boot_gp)]
  valid_gr <- boot_gr[is.finite(boot_gr)]
  p_gp <- if (length(valid_gp)) (1 + sum(valid_gp >= observed$GP)) / (1 + length(valid_gp)) else NA_real_
  p_gr <- if (length(valid_gr)) (1 + sum(valid_gr >= observed$GR)) / (1 + length(valid_gr)) else NA_real_

  data.frame(
    condition = condition,
    condition_id = unname(condition_ids[condition]),
    n = n, p = p,
    target_r2 = if (condition == "P0_null") NA_real_ else target_r2,
    replication = replication,
    seed = seed,
    n_boot = n_boot,
    observed_GP = observed$GP,
    observed_GR = observed$GR,
    null_GP_mean = safe_mean(valid_gp),
    null_GP_q95 = safe_quantile(valid_gp, 0.95),
    null_GR_mean = safe_mean(valid_gr),
    null_GR_q95 = safe_quantile(valid_gr, 0.95),
    p_GP = p_gp,
    p_GR = p_gr,
    reject_GP_05 = is.finite(p_gp) && p_gp <= ALPHA,
    reject_GR_05 = is.finite(p_gr) && p_gr <= ALPHA,
    stringsAsFactors = FALSE
  )
}

summarize_inference <- function(results) {
  groups <- unique(results[, c("condition", "condition_id", "n", "p", "target_r2"), drop = FALSE])
  rows <- vector("list", nrow(groups))
  for (i in seq_len(nrow(groups))) {
    g <- groups[i, , drop = FALSE]
    keep <- results$condition == g$condition[1] & results$n == g$n[1] & results$p == g$p[1]
    if (is.na(g$target_r2[1])) keep <- keep & is.na(results$target_r2)
    else keep <- keep & results$target_r2 == g$target_r2[1]
    sub <- results[keep, , drop = FALSE]
    nvalid_gp <- sum(is.finite(sub$p_GP))
    nvalid_gr <- sum(is.finite(sub$p_GR))
    xgp <- sum(sub$reject_GP_05, na.rm = TRUE)
    xgr <- sum(sub$reject_GR_05, na.rm = TRUE)
    cigp <- wilson_interval(xgp, nvalid_gp)
    cigr <- wilson_interval(xgr, nvalid_gr)
    rows[[i]] <- data.frame(
      condition = g$condition[1], condition_id = g$condition_id[1],
      n = g$n[1], p = g$p[1], target_r2 = g$target_r2[1],
      replications = nrow(sub),
      observed_GP_mean = safe_mean(sub$observed_GP),
      observed_GR_mean = safe_mean(sub$observed_GR),
      rejection_rate_GP = if (nvalid_gp) xgp / nvalid_gp else NA_real_,
      rejection_GP_ci_lower = cigp[1], rejection_GP_ci_upper = cigp[2],
      rejection_rate_GR = if (nvalid_gr) xgr / nvalid_gr else NA_real_,
      rejection_GR_ci_lower = cigr[1], rejection_GR_ci_upper = cigr[2],
      mean_p_GP = safe_mean(sub$p_GP), mean_p_GR = safe_mean(sub$p_GR),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

run_linear_null_inference <- function() {
  log_message("[Track B] Linear-null inference started.")
  all_blocks <- list()
  block_ix <- 1L
  track_total_tasks <- length(INFER_N_VALUES) * length(condition_levels) * INFER_REPLICATIONS
  track_completed_tasks <- 0L
  total_blocks <- length(INFER_N_VALUES) * length(condition_levels)
  current_block <- 0L

  for (n0 in INFER_N_VALUES) {
    for (cond0 in condition_levels) {
      current_block <- current_block + 1L
      checkpoint <- file.path(
        CHECKPOINT_OUTPUT,
        paste0("trackB_inference_n", n0, "_", unname(condition_ids[cond0]), ".rds")
      )
      expected <- INFER_REPLICATIONS
      if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
        tmp <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
        if (!is.null(tmp) && nrow(tmp) == expected) {
          track_completed_tasks <- track_completed_tasks + expected
          log_message("[Track B] Reusing checkpoint for n=", n0, ", condition=", cond0,
                      " | block ", current_block, "/", total_blocks,
                      " | Track B overall ", track_completed_tasks, "/",
                      track_total_tasks, " (",
                      sprintf("%.1f", 100 * track_completed_tasks / track_total_tasks), "% datasets)")
          all_blocks[[block_ix]] <- tmp
          block_ix <- block_ix + 1L
          next
        }
      }

      grid <- data.frame(
        condition = rep(cond0, INFER_REPLICATIONS),
        replication = seq_len(INFER_REPLICATIONS),
        n = n0,
        p = INFER_P,
        target_r2 = INFER_R2,
        stringsAsFactors = FALSE
      )

      res <- parallel_rows(grid, function(z) {
        run_linear_null_bootstrap_rep(
          condition = z$condition[1], n = as.integer(z$n[1]), p = as.integer(z$p[1]),
          target_r2 = as.numeric(z$target_r2[1]), replication = as.integer(z$replication[1]),
          n_boot = INFER_BOOT_B, span = ACE_SPAN
        )
      }, paste0("Track B block ", current_block, "/", total_blocks,
                     " n=", n0, ", condition=", cond0),
      batch_size = PROGRESS_BATCH_INFERENCE,
      overall_completed_before = track_completed_tasks,
      overall_total = track_total_tasks,
      overall_label = "Track B overall")

      track_completed_tasks <- track_completed_tasks + nrow(res)
      saveRDS(res, checkpoint)
      all_blocks[[block_ix]] <- res
      block_ix <- block_ix + 1L
    }
  }

  results <- do.call(rbind, all_blocks)
  summary <- summarize_inference(results)
  write.csv(results, file.path(SIM_OUTPUT, "B_linear_null_bootstrap_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "B_linear_null_bootstrap_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  pdf(file.path(SIM_OUTPUT, "B_rejection_rates.pdf"), width = 9, height = 6, onefile = TRUE)
  for (metric in c("rejection_rate_GP", "rejection_rate_GR")) {
    plot(NA, xlim = range(INFER_N_VALUES), ylim = c(0, 1),
         xlab = "n", ylab = "Rejection rate",
         main = paste0("Linear-null bootstrap: ", metric))
    for (cond in condition_levels) {
      d <- summary[summary$condition == cond, , drop = FALSE]
      d <- d[order(d$n), , drop = FALSE]
      lines(d$n, d[[metric]], type = "b",
            pch = match(cond, condition_levels),
            lty = match(cond, condition_levels))
    }
    abline(h = ALPHA, lty = 2)
    legend("right", legend = condition_ids[condition_levels],
           pch = seq_along(condition_levels), lty = seq_along(condition_levels),
           bty = "n", cex = 0.8)
  }
  dev.off()

  log_message("[Track B] Linear-null inference completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 12. TRACK C1: Correlated predictor robustness
# ----------------------------------------------------------------------------
run_correlation_robustness <- function() {
  log_message("[Track C1] Correlated-predictor robustness started.")
  checkpoint <- file.path(CHECKPOINT_OUTPUT, "trackC1_correlation.rds")
  expected <- length(RHO_VALUES) * length(condition_levels) * ROBUST_B

  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    results <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(results) || nrow(results) != expected) results <- NULL
  } else results <- NULL

  if (is.null(results)) {
    grid <- expand.grid(rho = RHO_VALUES, condition = condition_levels,
                        replication = seq_len(ROBUST_B), stringsAsFactors = FALSE)
    results <- parallel_rows(grid, function(z) {
      run_core_replication(
        condition = z$condition[1], n = ROBUST_N, p = ROBUST_P,
        replication = as.integer(z$replication[1]), target_r2 = ROBUST_R2,
        rho = as.numeric(z$rho[1]), error_dist = "normal", span = ACE_SPAN,
        k = CF_FOLDS, standardize_x = SIM_STANDARDIZE_X, seed_block = 300000000L
      )
    }, "Track C1 correlation robustness",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(results, checkpoint)
  } else log_message("[Track C1] Reusing checkpoint.")

  summary <- summarize_mc(results, c("condition", "condition_id", "rho"))
  write.csv(results, file.path(SIM_OUTPUT, "C1_correlated_predictors_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "C1_correlated_predictors_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  log_message("[Track C1] Correlated-predictor robustness completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 13. TRACK C2: ACE span sensitivity
# ----------------------------------------------------------------------------
run_span_sensitivity <- function() {
  log_message("[Track C2] ACE span sensitivity started.")
  checkpoint <- file.path(CHECKPOINT_OUTPUT, "trackC2_span.rds")
  expected <- length(SPAN_VALUES) * length(condition_levels) * SPAN_B

  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    results <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(results) || nrow(results) != expected) results <- NULL
  } else results <- NULL

  if (is.null(results)) {
    grid <- expand.grid(span = SPAN_VALUES, condition = condition_levels,
                        replication = seq_len(SPAN_B), stringsAsFactors = FALSE)
    results <- parallel_rows(grid, function(z) {
      run_core_replication(
        condition = z$condition[1], n = ROBUST_N, p = ROBUST_P,
        replication = as.integer(z$replication[1]), target_r2 = ROBUST_R2,
        rho = 0, error_dist = "normal", span = as.numeric(z$span[1]),
        k = CF_FOLDS, standardize_x = SIM_STANDARDIZE_X, seed_block = 400000000L
      )
    }, "Track C2 span sensitivity",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(results, checkpoint)
  } else log_message("[Track C2] Reusing checkpoint.")

  summary <- summarize_mc(results, c("condition", "condition_id", "span"))
  write.csv(results, file.path(SIM_OUTPUT, "C2_span_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "C2_span_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  log_message("[Track C2] ACE span sensitivity completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 14. TRACK C3: Error-distribution robustness
# ----------------------------------------------------------------------------
run_error_robustness <- function() {
  log_message("[Track C3] Error-distribution robustness started.")
  checkpoint <- file.path(CHECKPOINT_OUTPUT, "trackC3_error.rds")
  expected <- length(ERROR_DISTRIBUTIONS) * length(condition_levels) * ERROR_B

  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    results <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(results) || nrow(results) != expected) results <- NULL
  } else results <- NULL

  if (is.null(results)) {
    grid <- expand.grid(error_dist = ERROR_DISTRIBUTIONS, condition = condition_levels,
                        replication = seq_len(ERROR_B), stringsAsFactors = FALSE)
    results <- parallel_rows(grid, function(z) {
      run_core_replication(
        condition = z$condition[1], n = ROBUST_N, p = ROBUST_P,
        replication = as.integer(z$replication[1]), target_r2 = ROBUST_R2,
        rho = 0, error_dist = z$error_dist[1], span = ACE_SPAN,
        k = CF_FOLDS, standardize_x = SIM_STANDARDIZE_X, seed_block = 500000000L
      )
    }, "Track C3 error robustness",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(results, checkpoint)
  } else log_message("[Track C3] Reusing checkpoint.")

  summary <- summarize_mc(results, c("condition", "condition_id", "error_dist"))
  write.csv(results, file.path(SIM_OUTPUT, "C3_error_distribution_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "C3_error_distribution_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  log_message("[Track C3] Error-distribution robustness completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 15. TRACK C4: K-fold sensitivity
# ----------------------------------------------------------------------------
run_kfold_sensitivity <- function() {
  log_message("[Track C4] K-fold sensitivity started.")
  checkpoint <- file.path(CHECKPOINT_OUTPUT, "trackC4_kfold.rds")
  expected <- length(KFOLD_VALUES) * length(condition_levels) * KFOLD_B

  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    results <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(results) || nrow(results) != expected) results <- NULL
  } else results <- NULL

  if (is.null(results)) {
    grid <- expand.grid(k = KFOLD_VALUES, condition = condition_levels,
                        replication = seq_len(KFOLD_B), stringsAsFactors = FALSE)
    results <- parallel_rows(grid, function(z) {
      run_core_replication(
        condition = z$condition[1], n = ROBUST_N, p = ROBUST_P,
        replication = as.integer(z$replication[1]), target_r2 = ROBUST_R2,
        rho = 0, error_dist = "normal", span = ACE_SPAN,
        k = as.integer(z$k[1]), standardize_x = SIM_STANDARDIZE_X,
        seed_block = 600000000L
      )
    }, "Track C4 K-fold sensitivity",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(results, checkpoint)
  } else log_message("[Track C4] Reusing checkpoint.")

  summary <- summarize_mc(results, c("condition", "condition_id", "k"))
  write.csv(results, file.path(SIM_OUTPUT, "C4_kfold_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "C4_kfold_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  log_message("[Track C4] K-fold sensitivity completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 16. TRACK D: Benchmark methods
# ----------------------------------------------------------------------------
fit_gam_predict <- function(x_train, y_train, x_test) {
  train_df <- data.frame(y = y_train, x_train, check.names = FALSE)
  test_df <- data.frame(x_test, check.names = FALSE)
  xnames <- names(train_df)[-1]
  terms <- paste0("s(`", xnames, "`, k=8, bs='tp')")
  form <- stats::as.formula(paste("y ~", paste(terms, collapse = " + ")))
  fit <- mgcv::gam(form, data = train_df, method = "REML")
  as.numeric(stats::predict(fit, newdata = test_df))
}

fit_rf_predict <- function(x_train, y_train, x_test, seed) {
  train_df <- data.frame(y = y_train, x_train, check.names = FALSE)
  test_df <- data.frame(x_test, check.names = FALSE)
  fit <- ranger::ranger(
    y ~ ., data = train_df,
    num.trees = 300L,
    min.node.size = 5L,
    seed = seed,
    num.threads = 1L
  )
  as.numeric(stats::predict(fit, data = test_df)$predictions)
}

crossfit_benchmark <- function(x, y, k = 2L, span = ACE_SPAN,
                               seed = 1L, standardize_x = FALSE) {
  fold_id <- make_fold_id(nrow(x), k, seed)
  rows <- vector("list", k)
  for (fold in seq_len(k)) {
    te <- which(fold_id == fold)
    tr <- which(fold_id != fold)
    xtr_raw <- x[tr, , drop = FALSE]
    xte_raw <- x[te, , drop = FALSE]
    ytr <- y[tr]
    yte <- y[te]
    if (standardize_x) {
      sc <- standardize_train_test(xtr_raw, xte_raw)
      xtr <- sc$train
      xte <- sc$test
    } else {
      xtr <- xtr_raw
      xte <- xte_raw
    }

    lin <- stats::lm(y ~ ., data = data.frame(y = ytr, xtr, check.names = FALSE))
    pred_lin <- as.numeric(stats::predict(lin, newdata = data.frame(xte, check.names = FALSE)))

    ace_fit <- fit_ace_safely(xtr, ytr, span)
    ace_te <- apply_fitted_ace_to_test(ace_fit, xtr, ytr, xte, yte)

    pred_gam <- tryCatch(fit_gam_predict(xtr, ytr, xte), error = function(e) rep(NA_real_, length(yte)))
    pred_rf <- tryCatch(fit_rf_predict(xtr, ytr, xte, seed + fold), error = function(e) rep(NA_real_, length(yte)))

    rows[[fold]] <- data.frame(
      fold = fold,
      dCor_linear = if (all(is.finite(pred_lin))) distance_correlation(pred_lin, yte) else NA_real_,
      dCor_ACE = ace_te$R2,
      dCor_GAM = if (all(is.finite(pred_gam))) distance_correlation(pred_gam, yte) else NA_real_,
      dCor_RF = if (all(is.finite(pred_rf))) distance_correlation(pred_rf, yte) else NA_real_,
      stringsAsFactors = FALSE
    )
  }
  tab <- do.call(rbind, rows)
  c(dCor_linear = safe_mean(tab$dCor_linear),
    dCor_ACE = safe_mean(tab$dCor_ACE),
    dCor_GAM = safe_mean(tab$dCor_GAM),
    dCor_RF = safe_mean(tab$dCor_RF))
}

run_method_benchmark <- function() {
  log_message("[Track D] Method benchmark started.")
  checkpoint <- file.path(CHECKPOINT_OUTPUT, "trackD_benchmark.rds")
  expected <- length(condition_levels) * BENCHMARK_B

  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    results <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(results) || nrow(results) != expected) results <- NULL
  } else results <- NULL

  if (is.null(results)) {
    grid <- expand.grid(condition = condition_levels,
                        replication = seq_len(BENCHMARK_B), stringsAsFactors = FALSE)
    results <- parallel_rows(grid, function(z) {
      cond <- z$condition[1]
      repi <- as.integer(z$replication[1])
      seed <- replication_seed(cond, ROBUST_N, ROBUST_P, repi, ROBUST_R2, 700000000L)
      dat <- generate_data(cond, ROBUST_N, ROBUST_P, ROBUST_R2, 0, "normal", seed)
      t0 <- proc.time()[3]
      met <- crossfit_benchmark(dat$x, dat$y, 2L, ACE_SPAN, seed + 9L, SIM_STANDARDIZE_X)
      data.frame(condition = cond, condition_id = unname(condition_ids[cond]),
                 n = ROBUST_N, p = ROBUST_P, target_r2 = if (cond == "P0_null") NA_real_ else ROBUST_R2,
                 replication = repi,
                 dCor_linear = met["dCor_linear"], dCor_ACE = met["dCor_ACE"],
                 dCor_GAM = met["dCor_GAM"], dCor_RF = met["dCor_RF"],
                 runtime_seconds = as.numeric(proc.time()[3] - t0),
                 stringsAsFactors = FALSE)
    }, "Track D benchmark",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(results, checkpoint)
  } else log_message("[Track D] Reusing checkpoint.")

  methods <- c("dCor_linear", "dCor_ACE", "dCor_GAM", "dCor_RF")
  rows <- list(); ix <- 1L
  for (cond in condition_levels) {
    d <- results[results$condition == cond, , drop = FALSE]
    for (m in methods) {
      rows[[ix]] <- data.frame(condition = cond, condition_id = unname(condition_ids[cond]),
                               method = m, mean = safe_mean(d[[m]]), sd = safe_sd(d[[m]]),
                               se = safe_se(d[[m]]), stringsAsFactors = FALSE)
      ix <- ix + 1L
    }
  }
  summary <- do.call(rbind, rows)
  write.csv(results, file.path(SIM_OUTPUT, "D_method_benchmark_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(SIM_OUTPUT, "D_method_benchmark_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  log_message("[Track D] Method benchmark completed.")
  list(results = results, summary = summary)
}

# ----------------------------------------------------------------------------
# 17. UCI Concrete data loading
# ----------------------------------------------------------------------------
CONCRETE_URL <- paste0(
  "https://archive.ics.uci.edu/static/public/165/",
  "concrete+compressive+strength.zip"
)

load_concrete_data <- function(output_dir = REAL_OUTPUT) {
  data_dir <- file.path(output_dir, "uci_data")
  if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)
  zip_path <- file.path(data_dir, "concrete_compressive_strength.zip")
  unzip_dir <- file.path(data_dir, "unzipped")
  if (!dir.exists(unzip_dir)) dir.create(unzip_dir, recursive = TRUE)

  if (!file.exists(zip_path)) {
    log_message("Downloading UCI Concrete dataset.")
    utils::download.file(CONCRETE_URL, zip_path, mode = "wb", quiet = FALSE)
  }
  utils::unzip(zip_path, exdir = unzip_dir, overwrite = TRUE)

  files <- list.files(unzip_dir, recursive = TRUE, full.names = TRUE)
  candidates <- files[grepl("\\.xls[x]?$", files, ignore.case = TRUE)]
  if (length(candidates) == 0L) stop("No Excel file found in UCI Concrete archive.")
  preferred <- candidates[grepl("Concrete_Data", basename(candidates), ignore.case = TRUE)]
  xls_path <- if (length(preferred)) preferred[1] else candidates[1]

  raw <- as.data.frame(readxl::read_excel(xls_path), check.names = FALSE)
  if (ncol(raw) != 9L) stop("Expected 9 columns, found ", ncol(raw), ".")
  names(raw) <- c(
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age", "CompressiveStrength"
  )
  dat <- as.data.frame(lapply(raw, as.numeric), check.names = FALSE)
  dat <- dat[stats::complete.cases(dat), , drop = FALSE]
  if (nrow(dat) < 50L) stop("Too few complete observations in Concrete data.")
  dat
}

# ----------------------------------------------------------------------------
# 18. Real-data helper analyses
# ----------------------------------------------------------------------------
descriptive_table <- function(dat) {
  data.frame(
    variable = names(dat),
    n = vapply(dat, function(z) sum(is.finite(z)), numeric(1)),
    mean = vapply(dat, function(z) mean(z, na.rm = TRUE), numeric(1)),
    sd = vapply(dat, function(z) stats::sd(z, na.rm = TRUE), numeric(1)),
    min = vapply(dat, function(z) min(z, na.rm = TRUE), numeric(1)),
    q25 = vapply(dat, function(z) safe_quantile(z, 0.25), numeric(1)),
    median = vapply(dat, function(z) stats::median(z, na.rm = TRUE), numeric(1)),
    q75 = vapply(dat, function(z) safe_quantile(z, 0.75), numeric(1)),
    max = vapply(dat, function(z) max(z, na.rm = TRUE), numeric(1)),
    row.names = NULL
  )
}

univariate_dependence_table <- function(dat, x_cols, y_col) {
  y <- dat[[y_col]]
  rows <- lapply(x_cols, function(v) {
    x <- dat[[v]]
    data.frame(
      predictor = v,
      pearson_r = stats::cor(x, y, method = "pearson"),
      spearman_rho = stats::cor(x, y, method = "spearman"),
      dCor = distance_correlation(x, y),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

real_bootstrap_once <- function(x, y, fold_id, span, bootstrap_type, seed) {
  # The null is linear sufficiency. Predictor scaling is based on full data here
  # only for constructing the null fitted values; cross-fit evaluation below
  # still uses training-fold scaling.
  sx <- standardize_full(x)$x
  lin_fit <- stats::lm(y ~ ., data = data.frame(y = y, sx, check.names = FALSE))
  fitted0 <- as.numeric(stats::fitted(lin_fit))
  resid0 <- as.numeric(stats::residuals(lin_fit))
  resid0 <- resid0 - mean(resid0)

  set.seed(seed)
  if (bootstrap_type == "residual") {
    e_star <- sample(resid0, length(resid0), replace = TRUE)
  } else if (bootstrap_type == "wild_rademacher") {
    w <- sample(c(-1, 1), length(resid0), replace = TRUE)
    e_star <- resid0 * w
  } else {
    stop("Unknown bootstrap_type: ", bootstrap_type)
  }
  y_star <- fitted0 + e_star
  cf <- crossfit_metrics(x, y_star, CF_FOLDS, span,
                         fold_id = fold_id, standardize_x = REAL_STANDARDIZE_X)
  c(GP = cf$GP, GR = cf$GR, R0 = cf$R0, R1 = cf$R1, R2 = cf$R2, R3 = cf$R3)
}

run_real_linear_null_bootstrap <- function(x, y, output_dir, type, n_boot = REAL_BOOT_B) {
  fold_id <- make_fold_id(nrow(x), CF_FOLDS, BASE_SEED + 910000L)
  observed <- crossfit_metrics(x, y, CF_FOLDS, ACE_SPAN,
                               fold_id = fold_id, standardize_x = REAL_STANDARDIZE_X)

  checkpoint <- file.path(CHECKPOINT_OUTPUT, paste0("real_bootstrap_", type, "_B", n_boot, ".rds"))
  if (RESUME_FROM_CHECKPOINTS && file.exists(checkpoint)) {
    boot <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(boot) || nrow(boot) != n_boot) boot <- NULL
  } else boot <- NULL

  if (is.null(boot)) {
    grid <- data.frame(b = seq_len(n_boot))
    boot <- parallel_rows(grid, function(z) {
      b <- as.integer(z$b[1])
      met <- real_bootstrap_once(x, y, fold_id, ACE_SPAN, type,
                                 BASE_SEED + 920000L + b)
      data.frame(bootstrap_id = b,
                 GP = met["GP"], GR = met["GR"],
                 R0 = met["R0"], R1 = met["R1"], R2 = met["R2"], R3 = met["R3"],
                 stringsAsFactors = FALSE)
    }, paste0("Real-data ", type, " bootstrap"),
    batch_size = PROGRESS_BATCH_REAL_BOOT)
    saveRDS(boot, checkpoint)
  } else log_message("Reusing real-data bootstrap checkpoint: ", type)

  p_gp <- (1 + sum(boot$GP >= observed$GP, na.rm = TRUE)) / (1 + sum(is.finite(boot$GP)))
  p_gr <- (1 + sum(boot$GR >= observed$GR, na.rm = TRUE)) / (1 + sum(is.finite(boot$GR)))
  summary <- data.frame(
    bootstrap_type = type,
    statistic = c("GP", "GR"),
    observed = c(observed$GP, observed$GR),
    null_mean = c(safe_mean(boot$GP), safe_mean(boot$GR)),
    null_sd = c(safe_sd(boot$GP), safe_sd(boot$GR)),
    null_q95 = c(safe_quantile(boot$GP, 0.95), safe_quantile(boot$GR, 0.95)),
    p_value_one_sided = c(p_gp, p_gr),
    stringsAsFactors = FALSE
  )
  write.csv(boot, file.path(output_dir, paste0("bootstrap_", type, "_replications.csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(summary, file.path(output_dir, paste0("bootstrap_", type, "_summary.csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
  list(boot = boot, summary = summary, observed = observed)
}

# ----------------------------------------------------------------------------
# 19. TRACK E: Real-data analysis
# ----------------------------------------------------------------------------
run_concrete_analysis <- function() {
  log_message("[Track E] UCI Concrete real-data analysis started.")
  dat <- load_concrete_data(REAL_OUTPUT)
  y_col <- "CompressiveStrength"
  x_cols <- setdiff(names(dat), y_col)
  x <- as.matrix(dat[, x_cols, drop = FALSE])
  storage.mode(x) <- "double"
  y <- as.numeric(dat[[y_col]])

  write.csv(dat, file.path(REAL_OUTPUT, "concrete_cleaned_data.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(descriptive_table(dat), file.path(REAL_OUTPUT, "concrete_descriptive_statistics.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(stats::cor(dat), file.path(REAL_OUTPUT, "concrete_pearson_correlation_matrix.csv"),
            fileEncoding = "UTF-8")
  write.csv(univariate_dependence_table(dat, x_cols, y_col),
            file.path(REAL_OUTPUT, "concrete_univariate_dependence.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Full-sample descriptive stagewise analysis.
  ins <- insample_metrics(x, y, ACE_SPAN, REAL_STANDARDIZE_X)
  ins_tab <- data.frame(
    evaluation = "full_sample_descriptive",
    R0_standardized = ins$R0, R0_raw_units = ins$R0_raw,
    R1 = ins$R1, R2 = ins$R2, R3 = ins$R3,
    GP = ins$GP, GR = ins$GR,
    delta10 = ins$delta10, delta20 = ins$delta20, delta30 = ins$delta30,
    stringsAsFactors = FALSE
  )
  write.csv(ins_tab, file.path(REAL_OUTPUT, "concrete_insample_stagewise_results.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Repeated 2-fold cross-fitting; primary real-data descriptive result.
  cf_checkpoint <- file.path(CHECKPOINT_OUTPUT, paste0("real_cf_repeats_", REAL_CF_REPEATS, ".rds"))
  if (RESUME_FROM_CHECKPOINTS && file.exists(cf_checkpoint)) {
    cf_results <- tryCatch(readRDS(cf_checkpoint), error = function(e) NULL)
    if (is.null(cf_results) || nrow(cf_results) != REAL_CF_REPEATS) cf_results <- NULL
  } else cf_results <- NULL

  if (is.null(cf_results)) {
    grid <- data.frame(repeat_id = seq_len(REAL_CF_REPEATS))
    cf_results <- parallel_rows(grid, function(z) {
      r <- as.integer(z$repeat_id[1])
      cf <- crossfit_metrics(x, y, CF_FOLDS, ACE_SPAN,
                             seed = BASE_SEED + 10000L + r,
                             standardize_x = REAL_STANDARDIZE_X)
      data.frame(repeat_id = r,
                 R0 = cf$R0, R0_raw = cf$R0_raw,
                 R1 = cf$R1, R2 = cf$R2, R3 = cf$R3,
                 GP = cf$GP, GR = cf$GR,
                 delta10 = cf$delta10, delta20 = cf$delta20, delta30 = cf$delta30,
                 success_rate = cf$success_rate,
                 stringsAsFactors = FALSE)
    }, "Concrete repeated 2-fold cross-fitting",
    batch_size = PROGRESS_BATCH_STANDARD)
    saveRDS(cf_results, cf_checkpoint)
  } else log_message("[Track E] Reusing repeated-CF checkpoint.")

  metrics <- c("R0", "R0_raw", "R1", "R2", "R3", "GP", "GR", "delta10", "delta20", "delta30")
  cf_summary <- do.call(rbind, lapply(metrics, function(m) {
    z <- cf_results[[m]]
    data.frame(metric = m,
               mean = safe_mean(z), sd_across_splits = safe_sd(z),
               se_across_splits = safe_se(z),
               q025_across_splits = safe_quantile(z, 0.025),
               median_across_splits = safe_quantile(z, 0.5),
               q975_across_splits = safe_quantile(z, 0.975),
               stringsAsFactors = FALSE)
  }))
  write.csv(cf_results, file.path(REAL_OUTPUT, "concrete_repeated_crossfit_results.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(cf_summary, file.path(REAL_OUTPUT, "concrete_repeated_crossfit_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Full-sample linear coefficient table on standardized predictors.
  sx <- standardize_full(x)$x
  lm_full <- stats::lm(y ~ ., data = data.frame(y = y, sx, check.names = FALSE))
  ctab <- summary(lm_full)$coefficients
  coef_table <- data.frame(term = rownames(ctab), estimate = ctab[, 1],
                           std_error = ctab[, 2], t_value = ctab[, 3], p_value = ctab[, 4],
                           row.names = NULL)
  write.csv(coef_table, file.path(REAL_OUTPUT, "concrete_linear_model_standardized_coefficients.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # ACE transformations from standardized predictors, plotted against original units.
  ace_full <- ins$ace_fit
  if (ace_full$success) {
    pdf(file.path(REAL_OUTPUT, "concrete_ACE_predictor_transformations.pdf"),
        width = 8, height = 6, onefile = TRUE)
    for (j in seq_along(x_cols)) {
      ord <- order(x[, j])
      plot(x[ord, j], ace_full$tx[ord, j], pch = 16, cex = 0.4,
           xlab = x_cols[j], ylab = paste0("ACE phi_", j),
           main = paste0("ACE predictor transformation: ", x_cols[j]))
      lines(stats::lowess(x[ord, j], ace_full$tx[ord, j]), lwd = 2)
    }
    dev.off()

    pdf(file.path(REAL_OUTPUT, "concrete_ACE_response_transformation.pdf"), width = 8, height = 6)
    ord <- order(y)
    plot(y[ord], ace_full$ty[ord], pch = 16, cex = 0.4,
         xlab = "Concrete compressive strength",
         ylab = "ACE theta(Y)", main = "ACE response transformation")
    lines(stats::lowess(y[ord], ace_full$ty[ord]), lwd = 2)
    dev.off()
  }

  # Linear-null calibration: residual bootstrap + heteroskedasticity-robust wild sensitivity.
  residual_boot <- run_real_linear_null_bootstrap(x, y, REAL_OUTPUT, "residual", REAL_BOOT_B)
  wild_boot <- run_real_linear_null_bootstrap(x, y, REAL_OUTPUT, "wild_rademacher", REAL_BOOT_B)

  combined_boot <- rbind(residual_boot$summary, wild_boot$summary)
  write.csv(combined_boot, file.path(REAL_OUTPUT, "concrete_linear_null_bootstrap_combined_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  pdf(file.path(REAL_OUTPUT, "concrete_linear_null_bootstrap_histograms.pdf"),
      width = 8, height = 6, onefile = TRUE)
  for (obj in list(residual = residual_boot, wild_rademacher = wild_boot)) {
    nm <- obj$summary$bootstrap_type[1]
    hist(obj$boot$GP, breaks = "FD",
         main = paste0(nm, ": null distribution of GP"), xlab = "GP")
    abline(v = obj$observed$GP, lty = 2, lwd = 2)
    hist(obj$boot$GR, breaks = "FD",
         main = paste0(nm, ": null distribution of GR"), xlab = "GR")
    abline(v = obj$observed$GR, lty = 2, lwd = 2)
  }
  dev.off()

  # Real-data span sensitivity.
  span_grid <- expand.grid(span = SPAN_VALUES, repeat_id = seq_len(REAL_SENS_REPEATS),
                           stringsAsFactors = FALSE)
  real_span <- parallel_rows(span_grid, function(z) {
    cf <- crossfit_metrics(x, y, CF_FOLDS, as.numeric(z$span[1]),
                           seed = BASE_SEED + 30000L + as.integer(z$repeat_id[1]),
                           standardize_x = REAL_STANDARDIZE_X)
    data.frame(span = as.numeric(z$span[1]), repeat_id = as.integer(z$repeat_id[1]),
               R0 = cf$R0, R1 = cf$R1, R2 = cf$R2, R3 = cf$R3,
               GP = cf$GP, GR = cf$GR, stringsAsFactors = FALSE)
  }, "Concrete span sensitivity",
  batch_size = PROGRESS_BATCH_STANDARD)
  write.csv(real_span, file.path(REAL_OUTPUT, "concrete_span_sensitivity_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  real_span_summary <- aggregate(cbind(R0, R1, R2, R3, GP, GR) ~ span,
                                 data = real_span, FUN = mean)
  write.csv(real_span_summary, file.path(REAL_OUTPUT, "concrete_span_sensitivity_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Real-data K-fold sensitivity.
  k_grid <- expand.grid(k = KFOLD_VALUES, repeat_id = seq_len(REAL_SENS_REPEATS),
                        stringsAsFactors = FALSE)
  real_k <- parallel_rows(k_grid, function(z) {
    cf <- crossfit_metrics(x, y, as.integer(z$k[1]), ACE_SPAN,
                           seed = BASE_SEED + 40000L + as.integer(z$repeat_id[1]),
                           standardize_x = REAL_STANDARDIZE_X)
    data.frame(k = as.integer(z$k[1]), repeat_id = as.integer(z$repeat_id[1]),
               R0 = cf$R0, R1 = cf$R1, R2 = cf$R2, R3 = cf$R3,
               GP = cf$GP, GR = cf$GR, stringsAsFactors = FALSE)
  }, "Concrete K-fold sensitivity",
  batch_size = PROGRESS_BATCH_STANDARD)
  write.csv(real_k, file.path(REAL_OUTPUT, "concrete_kfold_sensitivity_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  real_k_summary <- aggregate(cbind(R0, R1, R2, R3, GP, GR) ~ k,
                              data = real_k, FUN = mean)
  write.csv(real_k_summary, file.path(REAL_OUTPUT, "concrete_kfold_sensitivity_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Same-fold method benchmark on Concrete data.
  bench_grid <- data.frame(repeat_id = seq_len(REAL_BENCH_REPEATS))
  real_bench <- parallel_rows(bench_grid, function(z) {
    r <- as.integer(z$repeat_id[1])
    met <- crossfit_benchmark(x, y, 2L, ACE_SPAN,
                              seed = BASE_SEED + 50000L + r,
                              standardize_x = REAL_STANDARDIZE_X)
    data.frame(repeat_id = r,
               dCor_linear = met["dCor_linear"], dCor_ACE = met["dCor_ACE"],
               dCor_GAM = met["dCor_GAM"], dCor_RF = met["dCor_RF"],
               stringsAsFactors = FALSE)
  }, "Concrete benchmark methods",
  batch_size = PROGRESS_BATCH_STANDARD)
  write.csv(real_bench, file.path(REAL_OUTPUT, "concrete_method_benchmark_replications.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  real_bench_summary <- data.frame(
    method = c("Linear", "ACE", "GAM", "RandomForest"),
    mean_dCor = c(safe_mean(real_bench$dCor_linear), safe_mean(real_bench$dCor_ACE),
                  safe_mean(real_bench$dCor_GAM), safe_mean(real_bench$dCor_RF)),
    sd_across_splits = c(safe_sd(real_bench$dCor_linear), safe_sd(real_bench$dCor_ACE),
                         safe_sd(real_bench$dCor_GAM), safe_sd(real_bench$dCor_RF)),
    stringsAsFactors = FALSE
  )
  write.csv(real_bench_summary, file.path(REAL_OUTPUT, "concrete_method_benchmark_summary.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  # Stagewise visualization using repeated-CF means.
  means <- setNames(cf_summary$mean, cf_summary$metric)
  pdf(file.path(REAL_OUTPUT, "concrete_stagewise_and_GPGR.pdf"), width = 8, height = 6, onefile = TRUE)
  barplot(c(R0 = means["R0"], R1 = means["R1"], R2 = means["R2"], R3 = means["R3"]),
          ylim = c(0, min(1, max(means[c("R0", "R1", "R2", "R3")], na.rm = TRUE) * 1.15)),
          ylab = "Distance correlation", main = "Concrete: repeated-CF R0--R3")
  plot(means["GP"], means["GR"], pch = 19,
       xlab = "Repeated-CF GP", ylab = "Repeated-CF GR",
       main = "Concrete: GP--GR diagnostic point",
       xlim = range(c(0, means["GP"] * 1.3), finite = TRUE),
       ylim = range(c(0, means["GR"] * 1.3), finite = TRUE))
  abline(h = 0, v = 0, lty = 2)
  text(means["GP"], means["GR"], labels = "Concrete", pos = 3)
  dev.off()

  run_info <- c(
    "Publication-scale UCI Concrete analysis",
    paste("URL:", CONCRETE_URL),
    paste("n:", nrow(x)),
    paste("p:", ncol(x)),
    paste("Predictors:", paste(x_cols, collapse = ", ")),
    paste("Response:", y_col),
    paste("ACE span:", ACE_SPAN),
    paste("Primary cross-fit folds:", CF_FOLDS),
    paste("Repeated CF:", REAL_CF_REPEATS),
    paste("Residual bootstrap B:", REAL_BOOT_B),
    paste("Wild bootstrap B:", REAL_BOOT_B),
    paste("Predictor standardization:", REAL_STANDARDIZE_X,
          "(training-fold mean/sd for cross-fitting)"),
    "",
    "Interpretation:",
    "GP = R2 - R1: predictor-side nonlinear representation gain.",
    "GR = R3 - R2: additional gain after transforming the response.",
    "Repeated-split quantiles describe split sensitivity; they are not formal sampling CIs.",
    "Residual/wild bootstrap p-values use a linear-sufficiency null and a fixed fold assignment.",
    "",
    capture.output(sessionInfo())
  )
  writeLines(run_info, file.path(REAL_OUTPUT, "concrete_run_info.txt"), useBytes = TRUE)

  log_message("[Track E] UCI Concrete real-data analysis completed.")
  list(data = dat, insample = ins_tab, cf_results = cf_results, cf_summary = cf_summary,
       residual_bootstrap = residual_boot$summary, wild_bootstrap = wild_boot$summary,
       span_summary = real_span_summary, kfold_summary = real_k_summary,
       benchmark_summary = real_bench_summary)
}

# ----------------------------------------------------------------------------
# 20. Save configuration before the long run
# ----------------------------------------------------------------------------
config_lines <- c(
  paste("RUN_PROFILE=", RUN_PROFILE),
  paste("OUTPUT_TAG=", OUTPUT_TAG),
  paste("ACEPACK_VERSION=", ACEPACK_VERSION),
  paste("BASE_SEED=", BASE_SEED),
  paste("REQUESTED_WORKERS=", REQUESTED_WORKERS),
  paste("MAX_WORKERS=", MAX_WORKERS),
  paste("SHOW_PROGRESS=", SHOW_PROGRESS),
  paste("PROGRESS_BATCH_MAIN=", PROGRESS_BATCH_MAIN),
  paste("PROGRESS_BATCH_STANDARD=", PROGRESS_BATCH_STANDARD),
  paste("PROGRESS_BATCH_INFERENCE=", PROGRESS_BATCH_INFERENCE),
  paste("PROGRESS_BATCH_REAL_BOOT=", PROGRESS_BATCH_REAL_BOOT),
  paste("ACE_SPAN=", ACE_SPAN),
  paste("CF_FOLDS=", CF_FOLDS),
  paste("MAIN_N_VALUES=", paste(MAIN_N_VALUES, collapse = ",")),
  paste("MAIN_P_VALUES=", paste(MAIN_P_VALUES, collapse = ",")),
  paste("MAIN_R2_VALUES=", paste(MAIN_R2_VALUES, collapse = ",")),
  paste("MAIN_B=", MAIN_B),
  paste("INFER_N_VALUES=", paste(INFER_N_VALUES, collapse = ",")),
  paste("INFER_P=", INFER_P),
  paste("INFER_R2=", INFER_R2),
  paste("INFER_REPLICATIONS=", INFER_REPLICATIONS),
  paste("INFER_BOOT_B=", INFER_BOOT_B),
  paste("ROBUST_B=", ROBUST_B),
  paste("SPAN_B=", SPAN_B),
  paste("ERROR_B=", ERROR_B),
  paste("KFOLD_B=", KFOLD_B),
  paste("BENCHMARK_B=", BENCHMARK_B),
  paste("REAL_CF_REPEATS=", REAL_CF_REPEATS),
  paste("REAL_BOOT_B=", REAL_BOOT_B),
  paste("REAL_SENS_REPEATS=", REAL_SENS_REPEATS),
  paste("REAL_BENCH_REPEATS=", REAL_BENCH_REPEATS),
  paste("RHO_VALUES=", paste(RHO_VALUES, collapse = ",")),
  paste("SPAN_VALUES=", paste(SPAN_VALUES, collapse = ",")),
  paste("ERROR_DISTRIBUTIONS=", paste(ERROR_DISTRIBUTIONS, collapse = ",")),
  paste("KFOLD_VALUES=", paste(KFOLD_VALUES, collapse = ",")),
  paste("SIM_STANDARDIZE_X=", SIM_STANDARDIZE_X),
  paste("REAL_STANDARDIZE_X=", REAL_STANDARDIZE_X)
)
writeLines(config_lines, file.path(ROOT_OUTPUT, "analysis_configuration.txt"), useBytes = TRUE)

# Save software/session information BEFORE any long work so diagnostics remain
# available even if the run is interrupted later.
writeLines(
  capture.output(sessionInfo()),
  file.path(ROOT_OUTPUT, "sessionInfo_preflight.txt"),
  useBytes = TRUE
)

# Mandatory preflight checks: never begin a multi-hour run with a broken ACE
# engine or a broken end-to-end cross-fitting pipeline.
ace_startup_selftest()
pipeline_startup_selftest()

preflight_lines <- c(
  paste0("timestamp=", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste0("R_version=", R.version.string),
  paste0("acepack_version=", ACEPACK_VERSION),
  paste0("ACE_SPAN=", ACE_SPAN),
  paste0("ACE_MAXIT=", ACE_MAXIT),
  paste0("ACE_SPAN_type=", typeof(as.double(ACE_SPAN))),
  paste0("ACE_MAXIT_type=", typeof(as.double(ACE_MAXIT))),
  paste0("workers=", MAX_WORKERS),
  "ACE_SELF_TEST=PASS",
  "PIPELINE_SELF_TEST=PASS"
)
writeLines(
  preflight_lines,
  file.path(ROOT_OUTPUT, "PREFLIGHT_PASS.txt"),
  useBytes = TRUE
)

# Print and save a publication-scale workload map before starting computation.
workload_plan <- print_workload_plan()

# ----------------------------------------------------------------------------
# 21. Execute requested analysis
# ----------------------------------------------------------------------------
real_result <- NULL
main_result <- NULL
infer_result <- NULL
corr_result <- NULL
span_result <- NULL
error_result <- NULL
kfold_result <- NULL
benchmark_result <- NULL

# Real data first: this produces the empirical application even if a later
# publication-scale simulation is interrupted.
if (DO_REAL_DATA && DO_REAL_DATA_FIRST) {
  real_result <- tryCatch(
    run_concrete_analysis(),
    error = function(e) {
      log_message("[Track E ERROR] ", conditionMessage(e))
      writeLines(c("Real-data analysis failed:", conditionMessage(e)),
                 file.path(REAL_OUTPUT, "REAL_DATA_ERROR.txt"))
      NULL
    }
  )
}

if (DO_MAIN_FACTORIAL) main_result <- run_main_factorial()
if (DO_LINEAR_NULL_INFERENCE) infer_result <- run_linear_null_inference()
if (DO_CORRELATION_ROBUSTNESS) corr_result <- run_correlation_robustness()
if (DO_SPAN_SENSITIVITY) span_result <- run_span_sensitivity()
if (DO_ERROR_ROBUSTNESS) error_result <- run_error_robustness()
if (DO_KFOLD_SENSITIVITY) kfold_result <- run_kfold_sensitivity()
if (DO_METHOD_BENCHMARK) benchmark_result <- run_method_benchmark()

if (DO_REAL_DATA && !DO_REAL_DATA_FIRST) {
  real_result <- tryCatch(
    run_concrete_analysis(),
    error = function(e) {
      log_message("[Track E ERROR] ", conditionMessage(e))
      writeLines(c("Real-data analysis failed:", conditionMessage(e)),
                 file.path(REAL_OUTPUT, "REAL_DATA_ERROR.txt"))
      NULL
    }
  )
}

# ----------------------------------------------------------------------------
# 22. Final manifest and console summary
# ----------------------------------------------------------------------------
manifest <- list.files(ROOT_OUTPUT, recursive = TRUE, full.names = FALSE)
writeLines(manifest, file.path(ROOT_OUTPUT, "output_manifest.txt"), useBytes = TRUE)

writeLines(c(
  "MAIN PAPER ANALYSIS COMPLETED",
  paste("Completed at:", Sys.time()),
  paste("Profile:", RUN_PROFILE),
  paste("Workers:", MAX_WORKERS),
  paste("Output:", normalizePath(ROOT_OUTPUT, winslash = "/", mustWork = FALSE)),
  "",
  "Primary paper statistics:",
  "R0 = dCor(X,Y)",
  "R1 = dCor(linear score,Y)",
  "R2 = dCor(ACE predictor score,Y)",
  "R3 = dCor(ACE predictor score,theta(Y))",
  "GP = R2-R1",
  "GR = R3-R2",
  "",
  "Interpret GP/GR as stagewise diagnostics, not orthogonal decomposition or causal effects.",
  "",
  capture.output(sessionInfo())
), file.path(ROOT_OUTPUT, "FINAL_SESSION_INFO.txt"), useBytes = TRUE)

log_message("All requested tracks completed.")
cat("\n============================================================\n")
cat("PAPER MAIN ANALYSIS COMPLETED\n")
cat("Output directory:\n")
cat(normalizePath(ROOT_OUTPUT, winslash = "/", mustWork = FALSE), "\n")
cat("See output_manifest.txt and run_log.txt.\n")
cat("============================================================\n")

# Reset future plan so RStudio is left in a clean state.
future::plan(future::sequential)


# ----------------------------------------------------------------------------
# 23. Publication-ready post-processing, QC, and additional outputs
# ----------------------------------------------------------------------------
# Why this section exists
# -----------------------
# The core tracks save detailed replication-level results.  For the paper we
# additionally want: (i) a hard validity/QC gate, (ii) compact main tables,
# (iii) empirical distribution summaries of GP/GR, (iv) representative
# diagnostic maps with Monte Carlo uncertainty, (v) size/power tables,
# (vi) compact robustness/benchmark tables, and (vii) one consolidated
# real-data result package.  This section reads the already-saved CSV files;
# it does NOT rerun simulations.

PAPER_OUTPUT <- file.path(ROOT_OUTPUT, "paper_outputs")
if (!dir.exists(PAPER_OUTPUT)) dir.create(PAPER_OUTPUT, recursive = TRUE)

read_csv_if_exists <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) {
      log_message("[Section 23 WARNING] Could not read ", basename(path),
                  ": ", conditionMessage(e))
      NULL
    }
  )
}

write_csv23 <- function(x, filename) {
  if (is.null(x) || !is.data.frame(x) || nrow(x) == 0L) return(FALSE)
  utils::write.csv(
    x, file.path(PAPER_OUTPUT, filename), row.names = FALSE, fileEncoding = "UTF-8"
  )
  TRUE
}

finite_rate23 <- function(x) {
  if (is.null(x) || length(x) == 0L) return(NA_real_)
  mean(is.finite(suppressWarnings(as.numeric(x))))
}

q23 <- function(x, prob) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  unname(stats::quantile(x, probs = prob, names = FALSE, type = 8))
}

# ---- 23.1 Hard QC gate -------------------------------------------------------
A_raw_path <- file.path(SIM_OUTPUT, "A_main_factorial_replications.csv")
A_sum_path <- file.path(SIM_OUTPUT, "A_main_factorial_summary.csv")
A_raw <- read_csv_if_exists(A_raw_path)
A_sum <- read_csv_if_exists(A_sum_path)

qc_rows <- list()
qci <- 1L
add_qc23 <- function(component, check, value, required, pass) {
  qc_rows[[qci]] <<- data.frame(
    component = component, check = check, value = as.character(value),
    required = as.character(required), pass = isTRUE(pass),
    stringsAsFactors = FALSE
  )
  qci <<- qci + 1L
}

add_qc23("Track A", "raw file exists", file.exists(A_raw_path), TRUE, file.exists(A_raw_path))
if (!is.null(A_raw)) {
  expected_A <- length(MAIN_N_VALUES) * length(MAIN_P_VALUES) *
    (1L + length(setdiff(condition_levels, "P0_null")) * length(MAIN_R2_VALUES)) * MAIN_B
  add_qc23("Track A", "row count", nrow(A_raw), expected_A, nrow(A_raw) == expected_A)
  add_qc23("Track A", "finite cf_R2 rate", sprintf("%.6f", finite_rate23(A_raw$cf_R2)),
           ">=0.95", is.finite(finite_rate23(A_raw$cf_R2)) && finite_rate23(A_raw$cf_R2) >= 0.95)
  add_qc23("Track A", "finite cf_R3 rate", sprintf("%.6f", finite_rate23(A_raw$cf_R3)),
           ">=0.95", is.finite(finite_rate23(A_raw$cf_R3)) && finite_rate23(A_raw$cf_R3) >= 0.95)
  add_qc23("Track A", "finite cf_GP rate", sprintf("%.6f", finite_rate23(A_raw$cf_GP)),
           ">=0.95", is.finite(finite_rate23(A_raw$cf_GP)) && finite_rate23(A_raw$cf_GP) >= 0.95)
  add_qc23("Track A", "finite cf_GR rate", sprintf("%.6f", finite_rate23(A_raw$cf_GR)),
           ">=0.95", is.finite(finite_rate23(A_raw$cf_GR)) && finite_rate23(A_raw$cf_GR) >= 0.95)
  if ("ace_success" %in% names(A_raw)) {
    asr <- mean(as.logical(A_raw$ace_success), na.rm = TRUE)
    add_qc23("Track A", "ACE success rate", sprintf("%.6f", asr), ">=0.95", is.finite(asr) && asr >= 0.95)
  }
  if ("cf_success" %in% names(A_raw)) {
    csr <- mean(as.logical(A_raw$cf_success), na.rm = TRUE)
    add_qc23("Track A", "cross-fit success rate", sprintf("%.6f", csr), ">=0.95", is.finite(csr) && csr >= 0.95)
  }
}

# Presence checks for the tracks that a full paper_main run is expected to produce.
expected_files23 <- c(
  "B inference" = file.path(SIM_OUTPUT, "B_linear_null_bootstrap_summary.csv"),
  "C1 correlation" = file.path(SIM_OUTPUT, "C1_correlated_predictors_summary.csv"),
  "C2 span" = file.path(SIM_OUTPUT, "C2_span_summary.csv"),
  "C3 error" = file.path(SIM_OUTPUT, "C3_error_distribution_summary.csv"),
  "C4 K-fold" = file.path(SIM_OUTPUT, "C4_kfold_summary.csv"),
  "D benchmark" = file.path(SIM_OUTPUT, "D_method_benchmark_summary.csv"),
  "E real-data CF" = file.path(REAL_OUTPUT, "concrete_repeated_crossfit_summary.csv"),
  "E real-data bootstrap" = file.path(REAL_OUTPUT, "concrete_linear_null_bootstrap_combined_summary.csv")
)
for (nm in names(expected_files23)) {
  add_qc23(nm, "summary file exists", file.exists(expected_files23[[nm]]), TRUE,
           file.exists(expected_files23[[nm]]))
}

qc23 <- do.call(rbind, qc_rows)
write_csv23(qc23, "00_publication_run_QC.csv")

qc_text <- c(
  "SECTION 23 PUBLICATION-RUN QC",
  paste("Generated:", Sys.time()),
  paste("acepack version:", ACEPACK_VERSION),
  "",
  capture.output(print(qc23, row.names = FALSE))
)
writeLines(qc_text, file.path(PAPER_OUTPUT, "00_publication_run_QC.txt"), useBytes = TRUE)

# Do not create apparently publication-ready GP/GR tables from an invalid ACE run.
primary_qc <- qc23$component == "Track A" &
  qc23$check %in% c("finite cf_R2 rate", "finite cf_R3 rate",
                    "finite cf_GP rate", "finite cf_GR rate")
if (!any(primary_qc) || any(!qc23$pass[primary_qc])) {
  fail23 <- paste0(
    "Section 23 stopped: Track A ACE metrics are not valid. ",
    "See paper_outputs/00_publication_run_QC.csv. ",
    "Do not use R2/R3/GP/GR from this run in the paper."
  )
  writeLines(fail23, file.path(PAPER_OUTPUT, "00_SECTION23_STOP_INVALID_ACE.txt"))
  log_message("[Section 23 STOP] ", fail23)
  stop(fail23)
}

# ---- 23.2 Compact main Monte Carlo tables -----------------------------------
if (!is.null(A_sum)) {
  wanted_A <- c(
    "condition", "condition_id", "n", "p", "target_r2", "replications",
    "ace_success_rate", "cf_success_rate",
    "cf_R0_mean", "cf_R0_se",
    "cf_R1_mean", "cf_R1_se",
    "cf_R2_mean", "cf_R2_se",
    "cf_R3_mean", "cf_R3_se",
    "cf_GP_mean", "cf_GP_se", "cf_GP_mc95_lower", "cf_GP_mc95_upper",
    "cf_GR_mean", "cf_GR_se", "cf_GR_mc95_lower", "cf_GR_mc95_upper",
    "cf_GP_mean_p0adj", "cf_GR_mean_p0adj",
    "runtime_seconds_mean"
  )
  wanted_A <- intersect(wanted_A, names(A_sum))
  A_compact <- A_sum[, wanted_A, drop = FALSE]
  write_csv23(A_compact, "01_main_factorial_compact_all_cells.csv")

  rep_keep <- A_sum$n == 500 & A_sum$p == 5 &
    (A_sum$condition == "P0_null" |
       (is.finite(A_sum$target_r2) & abs(A_sum$target_r2 - 0.5) < 1e-12))
  A_rep <- A_sum[rep_keep, wanted_A, drop = FALSE]
  ord <- match(A_rep$condition, condition_levels)
  A_rep <- A_rep[order(ord), , drop = FALSE]
  write_csv23(A_rep, "02_main_representative_n500_p5_R2_05.csv")
}

# Distribution shape matters for stagewise contrasts.  Means/SEs alone can
# hide skewness or instability, so save quantiles and Pr(gain > 0).
main_keys <- unique(A_raw[, c("condition", "condition_id", "n", "p", "target_r2"), drop = FALSE])
dist_rows <- vector("list", nrow(main_keys))
for (i in seq_len(nrow(main_keys))) {
  key <- main_keys[i, , drop = FALSE]
  keep <- A_raw$condition == key$condition[1] &
    A_raw$n == key$n[1] & A_raw$p == key$p[1]
  if (is.na(key$target_r2[1])) keep <- keep & is.na(A_raw$target_r2)
  else keep <- keep & abs(A_raw$target_r2 - key$target_r2[1]) < 1e-12
  d <- A_raw[keep, , drop = FALSE]
  gp <- d$cf_GP[is.finite(d$cf_GP)]
  gr <- d$cf_GR[is.finite(d$cf_GR)]
  dist_rows[[i]] <- data.frame(
    condition = key$condition[1], condition_id = key$condition_id[1],
    n = key$n[1], p = key$p[1], target_r2 = key$target_r2[1],
    replications = nrow(d),
    finite_GP_rate = length(gp) / nrow(d),
    GP_mean = safe_mean(gp), GP_sd = safe_sd(gp),
    GP_q025 = q23(gp, 0.025), GP_median = q23(gp, 0.5), GP_q975 = q23(gp, 0.975),
    Pr_GP_gt_0 = if (length(gp)) mean(gp > 0) else NA_real_,
    finite_GR_rate = length(gr) / nrow(d),
    GR_mean = safe_mean(gr), GR_sd = safe_sd(gr),
    GR_q025 = q23(gr, 0.025), GR_median = q23(gr, 0.5), GR_q975 = q23(gr, 0.975),
    Pr_GR_gt_0 = if (length(gr)) mean(gr > 0) else NA_real_,
    stringsAsFactors = FALSE
  )
}
A_dist <- do.call(rbind, dist_rows)
write_csv23(A_dist, "03_main_GP_GR_distribution_quantiles_and_sign.csv")

# Representative GP-GR diagnostic map with Monte Carlo 95% error bars.
if (!is.null(A_sum)) {
  dmap <- A_sum[A_sum$n == 500 & A_sum$p == 5 &
                  (A_sum$condition == "P0_null" |
                     (is.finite(A_sum$target_r2) & abs(A_sum$target_r2 - 0.5) < 1e-12)),
                , drop = FALSE]
  dmap <- dmap[is.finite(dmap$cf_GP_mean) & is.finite(dmap$cf_GR_mean), , drop = FALSE]
  if (nrow(dmap)) {
    pdf(file.path(PAPER_OUTPUT, "Figure23_01_GP_GR_map_n500_p5_R2_05.pdf"),
        width = 8, height = 7)
    xlo <- if ("cf_GP_mc95_lower" %in% names(dmap)) dmap$cf_GP_mc95_lower else dmap$cf_GP_mean
    xhi <- if ("cf_GP_mc95_upper" %in% names(dmap)) dmap$cf_GP_mc95_upper else dmap$cf_GP_mean
    ylo <- if ("cf_GR_mc95_lower" %in% names(dmap)) dmap$cf_GR_mc95_lower else dmap$cf_GR_mean
    yhi <- if ("cf_GR_mc95_upper" %in% names(dmap)) dmap$cf_GR_mc95_upper else dmap$cf_GR_mean
    xr <- range(c(0, xlo, xhi), finite = TRUE)
    yr <- range(c(0, ylo, yhi), finite = TRUE)
    plot(dmap$cf_GP_mean, dmap$cf_GR_mean, pch = 19, xlim = xr, ylim = yr,
         xlab = "Cross-fitted GP = R2 - R1",
         ylab = "Cross-fitted GR = R3 - R2",
         main = "Stagewise diagnostic map: n=500, p=5, target R-squared=0.5")
    abline(h = 0, v = 0, lty = 2)
    arrows(xlo, dmap$cf_GR_mean, xhi, dmap$cf_GR_mean,
           angle = 90, code = 3, length = 0.04)
    arrows(dmap$cf_GP_mean, ylo, dmap$cf_GP_mean, yhi,
           angle = 90, code = 3, length = 0.04)
    text(dmap$cf_GP_mean, dmap$cf_GR_mean, labels = dmap$condition_id, pos = 3)
    dev.off()
  }
}

# ---- 23.3 Linear-null size/power inference ----------------------------------
B_sum <- read_csv_if_exists(file.path(SIM_OUTPUT, "B_linear_null_bootstrap_summary.csv"))
if (!is.null(B_sum)) {
  wanted_B <- c(
    "condition", "condition_id", "n", "p", "target_r2", "replications",
    "observed_GP_mean", "observed_GR_mean",
    "rejection_rate_GP", "rejection_GP_ci_lower", "rejection_GP_ci_upper",
    "rejection_rate_GR", "rejection_GR_ci_lower", "rejection_GR_ci_upper",
    "mean_p_GP", "mean_p_GR"
  )
  B_compact <- B_sum[, intersect(wanted_B, names(B_sum)), drop = FALSE]
  if ("condition" %in% names(B_compact)) {
    B_compact$role <- ifelse(B_compact$condition %in% c("P0_null", "P1_linear"),
                             "null/size check", "nonlinear/power check")
  }
  write_csv23(B_compact, "04_linear_null_size_power_table.csv")

  pdf(file.path(PAPER_OUTPUT, "Figure23_02_linear_null_rejection_rates.pdf"),
      width = 9, height = 6, onefile = TRUE)
  for (met in c("rejection_rate_GP", "rejection_rate_GR")) {
    if (!met %in% names(B_sum)) next
    plot(NA, xlim = range(B_sum$n, finite = TRUE), ylim = c(0, 1),
         xlab = "Sample size n", ylab = "Rejection rate",
         main = paste0("Linear-null bootstrap: ", met))
    for (cond in condition_levels) {
      d <- B_sum[B_sum$condition == cond & is.finite(B_sum[[met]]), , drop = FALSE]
      if (!nrow(d)) next
      d <- d[order(d$n), , drop = FALSE]
      lines(d$n, d[[met]], type = "b",
            pch = match(cond, condition_levels), lty = match(cond, condition_levels))
    }
    abline(h = ALPHA, lty = 2)
    legend("right", legend = condition_ids[condition_levels],
           pch = seq_along(condition_levels), lty = seq_along(condition_levels),
           bty = "n", cex = 0.8)
  }
  dev.off()
}

# ---- 23.4 Compact robustness and benchmark outputs --------------------------
compact_summary23 <- function(filename, design_cols, outname) {
  d <- read_csv_if_exists(file.path(SIM_OUTPUT, filename))
  if (is.null(d)) return(NULL)
  wanted <- c("condition", "condition_id", design_cols,
              "replications", "ace_success_rate", "cf_success_rate",
              "cf_R1_mean", "cf_R2_mean", "cf_R3_mean",
              "cf_GP_mean", "cf_GP_se", "cf_GR_mean", "cf_GR_se")
  d2 <- d[, intersect(wanted, names(d)), drop = FALSE]
  write_csv23(d2, outname)
  d2
}

C1_23 <- compact_summary23("C1_correlated_predictors_summary.csv", "rho",
                           "05_robustness_correlated_predictors.csv")
C2_23 <- compact_summary23("C2_span_summary.csv", "span",
                           "06_robustness_span_sensitivity.csv")
C3_23 <- compact_summary23("C3_error_distribution_summary.csv", "error_dist",
                           "07_robustness_error_distribution.csv")
C4_23 <- compact_summary23("C4_kfold_summary.csv", "k",
                           "08_robustness_kfold_sensitivity.csv")

D_sum <- read_csv_if_exists(file.path(SIM_OUTPUT, "D_method_benchmark_summary.csv"))
if (!is.null(D_sum)) write_csv23(D_sum, "09_method_benchmark.csv")

# ---- 23.5 Consolidated UCI Concrete outputs ---------------------------------
E_ins <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_insample_stagewise_results.csv"))
E_cf <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_repeated_crossfit_summary.csv"))
E_boot <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_linear_null_bootstrap_combined_summary.csv"))
E_uni <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_univariate_dependence.csv"))
E_desc <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_descriptive_statistics.csv"))

if (!is.null(E_desc)) write_csv23(E_desc, "10_concrete_descriptive_statistics.csv")
if (!is.null(E_uni)) write_csv23(E_uni, "11_concrete_univariate_dependence.csv")
if (!is.null(E_boot)) write_csv23(E_boot, "13_concrete_linear_null_bootstrap.csv")

if (!is.null(E_cf)) {
  E_cf_out <- E_cf
  E_cf_out$interpretation_note <- paste0(
    "SD/quantiles are variation across repeated train-test splits; ",
    "they are not sampling confidence intervals."
  )
  write_csv23(E_cf_out, "12_concrete_repeated_crossfit_primary.csv")
}

# Real-data sensitivity tables
E_span <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_span_sensitivity_summary.csv"))
E_k <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_kfold_sensitivity_summary.csv"))
E_bench <- read_csv_if_exists(file.path(REAL_OUTPUT, "concrete_method_benchmark_summary.csv"))
if (!is.null(E_span)) write_csv23(E_span, "14_concrete_span_sensitivity.csv")
if (!is.null(E_k)) write_csv23(E_k, "15_concrete_kfold_sensitivity.csv")
if (!is.null(E_bench)) write_csv23(E_bench, "16_concrete_method_benchmark.csv")

# Compact stagewise real-data figure using repeated-CF means.
if (!is.null(E_cf) && all(c("metric", "mean") %in% names(E_cf))) {
  dR <- E_cf[E_cf$metric %in% c("R0", "R1", "R2", "R3") & is.finite(E_cf$mean), , drop = FALSE]
  dG <- E_cf[E_cf$metric %in% c("GP", "GR") & is.finite(E_cf$mean), , drop = FALSE]
  if (nrow(dR) || nrow(dG)) {
    pdf(file.path(PAPER_OUTPUT, "Figure23_03_concrete_primary_results.pdf"),
        width = 8, height = 6, onefile = TRUE)
    if (nrow(dR)) {
      dR <- dR[match(c("R0", "R1", "R2", "R3"), dR$metric), , drop = FALSE]
      dR <- dR[!is.na(dR$metric), , drop = FALSE]
      barplot(dR$mean, names.arg = dR$metric,
              ylab = "Repeated cross-fitted distance correlation",
              main = "Concrete data: stagewise dependence")
    }
    if (nrow(dG)) {
      dG <- dG[match(c("GP", "GR"), dG$metric), , drop = FALSE]
      dG <- dG[!is.na(dG$metric), , drop = FALSE]
      barplot(dG$mean, names.arg = dG$metric,
              ylab = "Stagewise gain",
              main = "Concrete data: GP and GR")
      abline(h = 0, lty = 2)
    }
    dev.off()
  }
}

# ---- 23.6 Output index / publication notes ----------------------------------
section23_manifest <- list.files(PAPER_OUTPUT, full.names = FALSE)
writeLines(section23_manifest, file.path(PAPER_OUTPUT, "SECTION23_OUTPUT_MANIFEST.txt"), useBytes = TRUE)

notes23 <- c(
  "SECTION 23: PAPER OUTPUT NOTES",
  paste("Generated:", Sys.time()),
  "",
  "Primary simulation reporting:",
  "- Use cross-fitted R0/R1/R2/R3 as descriptive stagewise dependence metrics.",
  "- Use GP=R2-R1 and GR=R3-R2 as stagewise diagnostic contrasts.",
  "- Do not interpret GP and GR as an orthogonal decomposition or causal effects.",
  "- Table 03 adds empirical quantiles and Pr(gain>0), which complement mean/MCSE.",
  "",
  "Inference:",
  "- Track B is the inferential size/power component for the linear-sufficiency null.",
  "- P0 and P1 are null/size checks; P2-P5 are nonlinear power/structure checks.",
  "",
  "Real data:",
  "- Repeated cross-fitting variation describes split sensitivity, not sampling uncertainty.",
  "- Residual/wild bootstrap p-values provide the linear-null calibration evidence.",
  "",
  "QC:",
  "- A publication run should have finite ACE metrics and high ACE/cross-fit success rates.",
  "- If 00_SECTION23_STOP_INVALID_ACE.txt exists, do not use that run for paper results."
)
writeLines(notes23, file.path(PAPER_OUTPUT, "SECTION23_REPORTING_NOTES.txt"), useBytes = TRUE)

# Refresh top-level manifest so Section 23 outputs are included.
manifest <- list.files(ROOT_OUTPUT, recursive = TRUE, full.names = FALSE)
writeLines(manifest, file.path(ROOT_OUTPUT, "output_manifest.txt"), useBytes = TRUE)

log_message("[Section 23] Publication-ready post-processing completed. Output: ",
            normalizePath(PAPER_OUTPUT, winslash = "/", mustWork = FALSE))
