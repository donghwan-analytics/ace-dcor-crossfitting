# =============================================================================
# P-to-One ACE--Distance Correlation
# EXTENDED MAIN SIMULATION: P6--P12
#
# Purpose
# -------
# This script adds the reviewer/advisor-oriented extended simulation only.
# It DOES NOT rerun the completed P0--P5 main simulation or the Concrete analysis.
#
# Final extended design
# ---------------------
#   n      = 200, 500, 1000
#   p      = 5, 10, 20
#   s      = 2, 5, 10, subject to s <= p
#   R^2    = 0.2, 0.5, 0.8
#   P6--P12
#   B      = 1000 per cell
#
# Valid (p,s) combinations:
#   (5,2), (5,5),
#   (10,2), (10,5), (10,10),
#   (20,2), (20,5), (20,10)
#
# Total number of Monte Carlo replications:
#   3(n) * 8(valid p,s) * 7(scenarios) * 3(R^2) * 1000
#   = 504,000
#
# Primary outputs:
#   R0, R1, R2, R3
#   GP = R2 - R1
#   GR = R3 - R2
# evaluated by 2-fold cross-fitting.
#
# IMPORTANT
# ---------
# - No pilot Monte Carlo is run.
# - Before the long run, mandatory unit/smoke tests check all P6--P12 x s cases.
# - Checkpointing is by (n,p,s,scenario), so an interruption does not require
#   restarting the entire 504,000-replication experiment.
# - Bootstrap inference is intentionally NOT applied to all 504 cells.
#   That would be computationally excessive. A representative inference track
#   can be run separately after the extended main results are inspected.
#
# Based on the validated v6 ACE/cross-fitting implementation.
# =============================================================================


# -----------------------------------------------------------------------------
# 0. Packages
# -----------------------------------------------------------------------------
required_packages <- c("acepack", "future", "future.apply")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

MIN_ACEPACK_VERSION <- package_version("1.6.1")
ACEPACK_VERSION <- utils::packageVersion("acepack")

if (ACEPACK_VERSION < MIN_ACEPACK_VERSION) {
  stop(
    paste0(
      "acepack >= ", MIN_ACEPACK_VERSION,
      " is required, but installed version is ", ACEPACK_VERSION, ".\n",
      "Run install.packages('acepack'), restart R/RStudio, and source again."
    )
  )
}

suppressPackageStartupMessages({
  library(acepack)
  library(future)
  library(future.apply)
})


# -----------------------------------------------------------------------------
# 1. Global configuration
# -----------------------------------------------------------------------------
BASE_SEED <- 20260904L
set.seed(BASE_SEED)

ACE_SPAN <- 0.5
# Must be DOUBLE, not 50L, for current acepack::set_control().
ACE_MAXIT <- 50

CF_FOLDS <- 2L
SIM_STANDARDIZE_X <- FALSE

EXT_N_VALUES <- c(200L, 500L, 1000L)
EXT_P_VALUES <- c(5L, 10L, 20L)
EXT_S_VALUES <- c(2L, 5L, 10L)
EXT_R2_VALUES <- c(0.2, 0.5, 0.8)
EXT_B <- 1000L

# Parallel execution: research-lab PC (Ryzen 7 8700F) target.
REQUESTED_WORKERS <- 8L
available_cores <- tryCatch(future::availableCores(), error = function(e) 1L)
MAX_WORKERS <- max(
  1L,
  min(REQUESTED_WORKERS, max(1L, as.integer(available_cores) - 1L))
)
if (!is.finite(MAX_WORKERS) || MAX_WORKERS < 1L) MAX_WORKERS <- 1L

# Prevent oversubscription inside each R worker.
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)

if (MAX_WORKERS > 1L) {
  future::plan(future::multisession, workers = MAX_WORKERS)
} else {
  future::plan(future::sequential)
}

options(future.globals.maxSize = 4 * 1024^3)

SHOW_PROGRESS <- TRUE
PROGRESS_BATCH_MAIN <- max(256L, MAX_WORKERS * 64L)

# Checkpoint/restart behavior.
RESUME_FROM_CHECKPOINTS <- TRUE

# If TRUE, run all preflight checks and then stop before 504,000 replications.
# Default FALSE = run the actual extended main simulation.
STOP_AFTER_PREFLIGHT <- FALSE


# -----------------------------------------------------------------------------
# 2. Project/output folders
# -----------------------------------------------------------------------------
PROJECT_DIR <- Sys.getenv("PORTFOLIO_PROJECT_DIR", unset = getwd())

if (!dir.exists(PROJECT_DIR)) {
  dir.create(PROJECT_DIR, recursive = TRUE, showWarnings = FALSE)
}
setwd(PROJECT_DIR)

OUTPUT_TAG <- "extended_P6_P12_v1"
ROOT_OUTPUT <- file.path(PROJECT_DIR, paste0("paper_", OUTPUT_TAG))
SIM_OUTPUT <- file.path(ROOT_OUTPUT, "simulation")
CHECKPOINT_OUTPUT <- file.path(ROOT_OUTPUT, "checkpoints")
PAPER_OUTPUT <- file.path(ROOT_OUTPUT, "paper_outputs")

for (d in c(ROOT_OUTPUT, SIM_OUTPUT, CHECKPOINT_OUTPUT, PAPER_OUTPUT)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

LOG_FILE <- file.path(ROOT_OUTPUT, "run_log.txt")

log_message <- function(...) {
  txt <- paste(..., collapse = "")
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", txt)
  cat(line, "\n")
  cat(line, "\n", file = LOG_FILE, append = TRUE)
}

cat("\n============================================================\n")
cat("EXTENDED MAIN SIMULATION: P6--P12\n")
cat("============================================================\n")
cat("Project :", normalizePath(PROJECT_DIR, winslash = "/", mustWork = FALSE), "\n")
cat("Output  :", normalizePath(ROOT_OUTPUT, winslash = "/", mustWork = FALSE), "\n")
cat("Workers :", MAX_WORKERS, "\n")
cat("============================================================\n\n")


# -----------------------------------------------------------------------------
# 3. Extended structural conditions
# -----------------------------------------------------------------------------
extended_condition_levels <- c(
  "P6_dense_quadratic_additive",
  "P7_dense_heterogeneous_additive",
  "P8_chain_pairwise_interaction",
  "P9_quadratic_single_index",
  "P10_threshold_additive",
  "P11_mixed_main_interaction",
  "P12_highorder_sign_interaction"
)

extended_condition_ids <- c("P6", "P7", "P8", "P9", "P10", "P11", "P12")
names(extended_condition_ids) <- extended_condition_levels

extended_condition_labels <- c(
  P6_dense_quadratic_additive =
    "P6: dense quadratic additive",
  P7_dense_heterogeneous_additive =
    "P7: dense heterogeneous additive",
  P8_chain_pairwise_interaction =
    "P8: chain pairwise interactions",
  P9_quadratic_single_index =
    "P9: quadratic single-index",
  P10_threshold_additive =
    "P10: threshold additive",
  P11_mixed_main_interaction =
    "P11: quadratic main + chain interactions",
  P12_highorder_sign_interaction =
    "P12: high-order sign interaction"
)

extended_condition_formulas <- c(
  P6_dense_quadratic_additive =
    "eta = s^(-1/2) sum_j (Xj^2-1)/sqrt(2)",
  P7_dense_heterogeneous_additive =
    "eta = s^(-1/2) sum_j f_j(Xj), standardized heterogeneous nonlinear f_j",
  P8_chain_pairwise_interaction =
    "eta = (s-1)^(-1/2) sum_{j=1}^{s-1} Xj*X(j+1)",
  P9_quadratic_single_index =
    "Z = s^(-1/2) sum_j Xj; eta=(Z^2-1)/sqrt(2)",
  P10_threshold_additive =
    "eta = s^(-1/2) sum_j {2 I(Xj>0)-1}",
  P11_mixed_main_interaction =
    "eta = (A+I)/sqrt(2); A=P6 component, I=P8 component",
  P12_highorder_sign_interaction =
    "eta = product_j {2 I(Xj>0)-1}"
)

extended_condition_expectation <- c(
  P6_dense_quadratic_additive =
    "GP expected large; GR expected relatively small",
  P7_dense_heterogeneous_additive =
    "GP expected positive/large; GR expected relatively small",
  P8_chain_pairwise_interaction =
    "interaction stress test; GR may become important",
  P9_quadratic_single_index =
    "mixed additive/nonadditive recovery expected",
  P10_threshold_additive =
    "non-smooth additive stress test; GP expected positive",
  P11_mixed_main_interaction =
    "both GP and GR can be positive",
  P12_highorder_sign_interaction =
    "adversarial high-order interaction; GP/GR may remain small despite raw dependence"
)


# Consistent scenario colors for publication-quality figures.
# Muted but distinguishable palette; line type and point shape are still kept
# so the figures remain interpretable in grayscale printing.
scenario_colors <- c(
  P6_dense_quadratic_additive     = "#1B9E77",
  P7_dense_heterogeneous_additive = "#D95F02",
  P8_chain_pairwise_interaction   = "#7570B3",
  P9_quadratic_single_index       = "#E7298A",
  P10_threshold_additive          = "#66A61E",
  P11_mixed_main_interaction      = "#E6AB02",
  P12_highorder_sign_interaction  = "#A6761D"
)

scenario_point_shapes <- c(
  P6_dense_quadratic_additive     = 16,
  P7_dense_heterogeneous_additive = 17,
  P8_chain_pairwise_interaction   = 15,
  P9_quadratic_single_index       = 18,
  P10_threshold_additive          = 0,
  P11_mixed_main_interaction      = 1,
  P12_highorder_sign_interaction  = 2
)

scenario_line_types <- c(
  P6_dense_quadratic_additive     = 1,
  P7_dense_heterogeneous_additive = 2,
  P8_chain_pairwise_interaction   = 3,
  P9_quadratic_single_index       = 4,
  P10_threshold_additive          = 5,
  P11_mixed_main_interaction      = 6,
  P12_highorder_sign_interaction  = 1
)

condition_definitions <- data.frame(
  condition = extended_condition_levels,
  condition_id = unname(extended_condition_ids[extended_condition_levels]),
  label = unname(extended_condition_labels[extended_condition_levels]),
  formula = unname(extended_condition_formulas[extended_condition_levels]),
  expected_diagnostic_profile =
    unname(extended_condition_expectation[extended_condition_levels]),
  stringsAsFactors = FALSE
)

write.csv(
  condition_definitions,
  file.path(SIM_OUTPUT, "F_condition_definitions_P6_P12.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)


# -----------------------------------------------------------------------------
# 4. Valid p-s combinations and workload
# -----------------------------------------------------------------------------
valid_ps <- expand.grid(
  p = EXT_P_VALUES,
  s = EXT_S_VALUES,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
valid_ps <- valid_ps[valid_ps$s <= valid_ps$p, , drop = FALSE]
valid_ps <- valid_ps[order(valid_ps$p, valid_ps$s), , drop = FALSE]
rownames(valid_ps) <- NULL

EXPECTED_REPLICATIONS <-
  length(EXT_N_VALUES) *
  nrow(valid_ps) *
  length(extended_condition_levels) *
  length(EXT_R2_VALUES) *
  EXT_B

if (EXPECTED_REPLICATIONS != 504000L) {
  stop(
    "Internal workload check failed. Expected 504,000 replications but got ",
    EXPECTED_REPLICATIONS, "."
  )
}

workload <- data.frame(
  item = c(
    "n values",
    "valid (p,s) combinations",
    "P6-P12 scenarios",
    "R2 values",
    "replications per cell",
    "TOTAL replications"
  ),
  value = c(
    length(EXT_N_VALUES),
    nrow(valid_ps),
    length(extended_condition_levels),
    length(EXT_R2_VALUES),
    EXT_B,
    EXPECTED_REPLICATIONS
  ),
  stringsAsFactors = FALSE
)

write.csv(
  valid_ps,
  file.path(ROOT_OUTPUT, "VALID_P_S_COMBINATIONS.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  workload,
  file.path(ROOT_OUTPUT, "WORKLOAD_PLAN_EXTENDED.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

log_message("Extended design:")
log_message("  n = ", paste(EXT_N_VALUES, collapse = ", "))
log_message("  p = ", paste(EXT_P_VALUES, collapse = ", "))
log_message("  s = ", paste(EXT_S_VALUES, collapse = ", "), " with s <= p")
log_message("  R2 = ", paste(EXT_R2_VALUES, collapse = ", "))
log_message("  scenarios = P6--P12")
log_message("  B = ", EXT_B)
log_message("  valid (p,s) combinations = ", nrow(valid_ps))
log_message("  TOTAL = ", format(EXPECTED_REPLICATIONS, big.mark = ","), " replications")


# -----------------------------------------------------------------------------
# 5. Generic numerical helpers
# -----------------------------------------------------------------------------
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
  if (length(x) == 0L) {
    NA_real_
  } else {
    unname(stats::quantile(x, probs = prob, names = FALSE, type = 7))
  }
}

r2_to_noise_sd <- function(target_r2) {
  if (!is.finite(target_r2) || target_r2 <= 0 || target_r2 >= 1) {
    stop("target_r2 must be strictly between 0 and 1.")
  }
  sqrt((1 - target_r2) / target_r2)
}

distance_correlation <- function(x, y) {
  x <- as.matrix(x)
  y <- as.matrix(y)

  if (nrow(x) != nrow(y)) {
    stop("x and y must have the same number of rows.")
  }

  n <- nrow(x)
  if (n < 2L) stop("At least two observations are required.")

  a <- as.matrix(stats::dist(x))
  b <- as.matrix(stats::dist(y))

  A <- a - rowMeans(a) -
    matrix(colMeans(a), nrow = n, ncol = n, byrow = TRUE) +
    mean(a)

  B <- b - rowMeans(b) -
    matrix(colMeans(b), nrow = n, ncol = n, byrow = TRUE) +
    mean(b)

  dcov2_xy <- max(mean(A * B), 0)
  dcov2_xx <- max(mean(A * A), 0)
  dcov2_yy <- max(mean(B * B), 0)

  denominator <- sqrt(sqrt(dcov2_xx * dcov2_yy))

  if (!is.finite(denominator) ||
      denominator <= .Machine$double.eps) {
    return(0)
  }

  sqrt(dcov2_xy) / denominator
}

standardize_train_test <- function(x_train, x_test) {
  mu <- colMeans(x_train)
  sc <- apply(x_train, 2, stats::sd)
  sc[!is.finite(sc) | sc <= .Machine$double.eps] <- 1

  train_z <- sweep(sweep(x_train, 2, mu, "-"), 2, sc, "/")
  test_z <- sweep(sweep(x_test, 2, mu, "-"), 2, sc, "/")

  list(train = train_z, test = test_z, center = mu, scale = sc)
}

standardize_full <- function(x) {
  mu <- colMeans(x)
  sc <- apply(x, 2, stats::sd)
  sc[!is.finite(sc) | sc <= .Machine$double.eps] <- 1

  z <- sweep(sweep(x, 2, mu, "-"), 2, sc, "/")
  list(x = z, center = mu, scale = sc)
}


# -----------------------------------------------------------------------------
# 6. Extended DGM: P6--P12
# -----------------------------------------------------------------------------

make_x_matrix_extended <- function(n, p) {
  x <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  colnames(x) <- paste0("X", seq_len(p))
  x
}

# Standardized nonlinear basis functions for N(0,1).
# Each function below has population mean 0 and variance 1.
h2 <- function(x) {
  (x^2 - 1) / sqrt(2)
}

h3 <- function(x) {
  (x^3 - 3 * x) / sqrt(6)
}

h_abs <- function(x) {
  mu <- sqrt(2 / pi)
  sd0 <- sqrt(1 - 2 / pi)
  (abs(x) - mu) / sd0
}

h_sin <- function(x) {
  # E sin(X)=0; Var sin(X) = (1-exp(-2))/2 for X~N(0,1).
  sd0 <- sqrt((1 - exp(-2)) / 2)
  sin(x) / sd0
}

h_cos <- function(x) {
  # E cos(X)=exp(-1/2)
  # E cos^2(X)=(1+exp(-2))/2
  mu <- exp(-0.5)
  var0 <- (1 + exp(-2)) / 2 - exp(-1)
  (cos(x) - mu) / sqrt(var0)
}

heterogeneous_component <- function(x, j) {
  type <- ((j - 1L) %% 5L) + 1L

  if (type == 1L) return(h2(x))
  if (type == 2L) return(h3(x))
  if (type == 3L) return(h_sin(x))
  if (type == 4L) return(h_abs(x))
  if (type == 5L) return(h_cos(x))

  stop("Unexpected heterogeneous component type.")
}

threshold_component <- function(x) {
  ifelse(x > 0, 1, -1)
}

raw_extended_signal <- function(x, condition, s) {
  x <- as.matrix(x)
  p <- ncol(x)

  if (!is.finite(s) || s < 2L || s > p) {
    stop("s must satisfy 2 <= s <= p.")
  }

  active <- x[, seq_len(s), drop = FALSE]

  # P6: dense quadratic additive.
  # Each h2(Xj) has mean 0 and variance 1.
  # Independence across active predictors gives Var(eta)=1.
  if (condition == "P6_dense_quadratic_additive") {
    terms <- apply(active, 2, h2)
    if (is.null(dim(terms))) terms <- matrix(terms, ncol = 1L)
    return(rowSums(terms) / sqrt(s))
  }

  # P7: dense heterogeneous additive.
  # Different standardized nonlinear functions are applied to active variables.
  if (condition == "P7_dense_heterogeneous_additive") {
    terms <- matrix(NA_real_, nrow = nrow(x), ncol = s)
    for (j in seq_len(s)) {
      terms[, j] <- heterogeneous_component(active[, j], j)
    }
    return(rowSums(terms) / sqrt(s))
  }

  # P8: chain pairwise interaction.
  # Cov(Xj Xj+1, Xk Xk+1)=0 for distinct chain terms under iid N(0,1),
  # hence division by sqrt(s-1) gives Var(eta)=1.
  if (condition == "P8_chain_pairwise_interaction") {
    out <- rep(0, nrow(x))
    for (j in seq_len(s - 1L)) {
      out <- out + active[, j] * active[, j + 1L]
    }
    return(out / sqrt(s - 1L))
  }

  # P9: quadratic single-index.
  # Z ~ N(0,1), so (Z^2-1)/sqrt(2) has mean 0 and variance 1.
  if (condition == "P9_quadratic_single_index") {
    z <- rowSums(active) / sqrt(s)
    return(h2(z))
  }

  # P10: additive non-smooth threshold structure.
  # Each sign component is Rademacher (+1/-1), mean 0, variance 1.
  if (condition == "P10_threshold_additive") {
    signs <- ifelse(active > 0, 1, -1)
    return(rowSums(signs) / sqrt(s))
  }

  # P11: mixture of P6-type main nonlinear effects and P8-type interactions.
  # The two components each have variance 1 and population covariance 0
  # under iid standard normal X, hence (A+I)/sqrt(2) has variance 1.
  if (condition == "P11_mixed_main_interaction") {
    main_terms <- apply(active, 2, h2)
    if (is.null(dim(main_terms))) main_terms <- matrix(main_terms, ncol = 1L)
    a <- rowSums(main_terms) / sqrt(s)

    inter <- rep(0, nrow(x))
    for (j in seq_len(s - 1L)) {
      inter <- inter + active[, j] * active[, j + 1L]
    }
    i_part <- inter / sqrt(s - 1L)

    return((a + i_part) / sqrt(2))
  }

  # P12: bounded high-order sign interaction (parity-type stress test).
  # eta is exactly +/-1 with population mean 0 and variance 1.
  # This avoids the extreme fourth moments of product_j Xj for large s.
  if (condition == "P12_highorder_sign_interaction") {
    signs <- ifelse(active > 0, 1, -1)
    return(apply(signs, 1, prod))
  }

  stop("Unknown extended condition: ", condition)
}

generate_extended_data <- function(
    condition,
    n,
    p,
    s,
    target_r2 = 0.5,
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)

  if (!(condition %in% extended_condition_levels)) {
    stop("Unknown extended condition: ", condition)
  }

  if (!(p %in% EXT_P_VALUES)) {
    stop("p must be one of: ", paste(EXT_P_VALUES, collapse = ", "))
  }

  if (!(s %in% EXT_S_VALUES) || s > p) {
    stop("Invalid active dimension s=", s, " for p=", p, ".")
  }

  x <- make_x_matrix_extended(n, p)
  eta <- raw_extended_signal(x, condition, s)

  if (any(!is.finite(eta))) {
    stop("Non-finite eta generated for ", condition, ", p=", p, ", s=", s)
  }

  noise_sd <- r2_to_noise_sd(target_r2)
  eps_std <- stats::rnorm(n)
  epsilon <- noise_sd * eps_std
  y <- eta + epsilon

  list(
    x = x,
    y = as.numeric(y),
    eta = as.numeric(eta),
    epsilon = as.numeric(epsilon),
    noise_sd = noise_sd,
    target_r2 = target_r2,
    sample_eta_mean = mean(eta),
    sample_eta_var = stats::var(eta),
    sample_epsilon_var = stats::var(epsilon),
    sample_y_var = stats::var(y),
    sample_signal_fraction =
      stats::var(eta) / (stats::var(eta) + stats::var(epsilon))
  )
}


# -----------------------------------------------------------------------------
# 7. ACE fitting and out-of-sample transformation
# -----------------------------------------------------------------------------
make_ace_control <- function(span = ACE_SPAN, maxit = ACE_MAXIT) {
  span <- as.double(span)
  maxit <- as.double(maxit)

  if (length(span) != 1L ||
      !is.finite(span) ||
      span < 0 ||
      span > 1) {
    stop("ACE span must be one finite numeric value in [0,1].")
  }

  if (length(maxit) != 1L ||
      !is.finite(maxit) ||
      maxit <= 0) {
    stop("ACE maxit must be one positive finite numeric value.")
  }

  if (!inherits(span, "numeric") ||
      !inherits(maxit, "numeric")) {
    stop("ACE control values must inherit from class 'numeric'.")
  }

  list(span = span, maxit = maxit)
}

fit_ace_safely <- function(x, y, span = ACE_SPAN) {
  tryCatch({
    t0 <- proc.time()[3]

    fit <- acepack::ace(
      x = x,
      y = y,
      delrsq = 1e-4,
      control = make_ace_control(
        span = span,
        maxit = ACE_MAXIT
      ),
      on.error = function(message) stop(message)
    )

    elapsed <- proc.time()[3] - t0

    tx <- as.matrix(fit$tx)
    ty <- as.numeric(fit$ty)
    score <- rowSums(tx)

    if (any(!is.finite(score)) ||
        any(!is.finite(ty))) {
      stop("ACE returned non-finite transformed values.")
    }

    list(
      success = TRUE,
      fit = fit,
      tx = tx,
      ty = ty,
      score = score,
      elapsed_seconds = as.numeric(elapsed),
      error_message = NA_character_
    )
  }, error = function(e) {
    list(
      success = FALSE,
      fit = NULL,
      tx = NULL,
      ty = NULL,
      score = NULL,
      elapsed_seconds = NA_real_,
      error_message = conditionMessage(e)
    )
  })
}

make_interp_function <- function(x_train, z_train) {
  ok <- is.finite(x_train) & is.finite(z_train)
  x_train <- x_train[ok]
  z_train <- z_train[ok]

  if (length(x_train) == 0L) {
    return(function(x_new) rep(0, length(x_new)))
  }

  if (length(x_train) < 4L ||
      !is.finite(stats::sd(z_train)) ||
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

apply_fitted_ace_to_test <- function(
    ace_result,
    x_train,
    y_train,
    x_test,
    y_test
) {
  if (!ace_result$success) {
    return(list(
      success = FALSE,
      R2 = NA_real_,
      R3 = NA_real_,
      score_test = NULL,
      ty_test = NULL,
      error_message = ace_result$error_message
    ))
  }

  p <- ncol(x_train)
  tx_test <- matrix(
    NA_real_,
    nrow = nrow(x_test),
    ncol = p
  )

  for (j in seq_len(p)) {
    fj <- make_interp_function(
      x_train[, j],
      ace_result$tx[, j]
    )
    tx_test[, j] <- fj(x_test[, j])
  }

  theta_fun <- make_interp_function(
    y_train,
    ace_result$ty
  )

  ty_test <- theta_fun(y_test)
  score_test <- rowSums(tx_test)

  if (any(!is.finite(score_test)) ||
      any(!is.finite(ty_test))) {
    return(list(
      success = FALSE,
      R2 = NA_real_,
      R3 = NA_real_,
      score_test = NULL,
      ty_test = NULL,
      error_message = "Non-finite test transformation."
    ))
  }

  list(
    success = TRUE,
    R2 = distance_correlation(score_test, y_test),
    R3 = distance_correlation(score_test, ty_test),
    score_test = score_test,
    ty_test = ty_test,
    error_message = NA_character_
  )
}

make_fold_id <- function(n, k = CF_FOLDS, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  fold_id <- rep(seq_len(k), length.out = n)
  sample(fold_id, size = n, replace = FALSE)
}


# -----------------------------------------------------------------------------
# 8. Stagewise metrics
# -----------------------------------------------------------------------------
transformation_variance_ratio <- function(ace_tx, active_s) {
  tx <- as.matrix(ace_tx)

  if (ncol(tx) <= active_s) return(0)

  tx_var <- apply(tx, 2, stats::var)
  total_var <- sum(tx_var, na.rm = TRUE)

  if (!is.finite(total_var) ||
      total_var <= .Machine$double.eps) {
    return(NA_real_)
  }

  inactive_idx <- seq.int(active_s + 1L, ncol(tx))
  sum(tx_var[inactive_idx], na.rm = TRUE) / total_var
}

insample_metrics <- function(
    x,
    y,
    active_s,
    span = ACE_SPAN,
    standardize_x = FALSE
) {
  x_raw <- as.matrix(x)

  if (standardize_x) {
    sx <- standardize_full(x_raw)
    x_use <- sx$x
  } else {
    x_use <- x_raw
  }

  R0_raw <- distance_correlation(x_raw, y)
  R0 <- if (standardize_x) {
    distance_correlation(x_use, y)
  } else {
    R0_raw
  }

  lin_df <- data.frame(y = y, x_use, check.names = FALSE)
  lin_fit <- stats::lm(y ~ ., data = lin_df)
  linear_score <- as.numeric(stats::fitted(lin_fit))
  R1 <- distance_correlation(linear_score, y)

  ace_fit <- fit_ace_safely(x_use, y, span)

  if (ace_fit$success) {
    R2 <- distance_correlation(ace_fit$score, y)
    R3 <- distance_correlation(ace_fit$score, ace_fit$ty)
    ace_rsq <- unname(ace_fit$fit$rsq)
    inactive_ratio <- transformation_variance_ratio(
      ace_fit$tx,
      active_s
    )
  } else {
    R2 <- NA_real_
    R3 <- NA_real_
    ace_rsq <- NA_real_
    inactive_ratio <- NA_real_
  }

  list(
    R0 = R0,
    R0_raw = R0_raw,
    R1 = R1,
    R2 = R2,
    R3 = R3,
    GP = R2 - R1,
    GR = R3 - R2,
    delta10 = R1 - R0,
    delta20 = R2 - R0,
    delta30 = R3 - R0,
    ace_rsq = ace_rsq,
    inactive_tx_variance_ratio = inactive_ratio,
    ace_success = ace_fit$success,
    ace_error_message = ace_fit$error_message,
    ace_fit = ace_fit,
    linear_fit = lin_fit
  )
}

crossfit_metrics <- function(
    x,
    y,
    active_s,
    k = CF_FOLDS,
    span = ACE_SPAN,
    seed = NULL,
    fold_id = NULL,
    standardize_x = FALSE
) {
  x <- as.matrix(x)

  if (is.null(fold_id)) {
    fold_id <- make_fold_id(nrow(x), k, seed)
  }

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
      sc <- standardize_train_test(
        x_train_raw,
        x_test_raw
      )
      x_train <- sc$train
      x_test <- sc$test
    } else {
      x_train <- x_train_raw
      x_test <- x_test_raw
    }

    R0_raw <- distance_correlation(
      x_test_raw,
      y_test
    )

    R0 <- if (standardize_x) {
      distance_correlation(x_test, y_test)
    } else {
      R0_raw
    }

    train_df <- data.frame(
      y = y_train,
      x_train,
      check.names = FALSE
    )

    test_df <- data.frame(
      x_test,
      check.names = FALSE
    )

    lin_fit <- stats::lm(
      y ~ .,
      data = train_df
    )

    lin_pred <- suppressWarnings(
      as.numeric(
        stats::predict(
          lin_fit,
          newdata = test_df
        )
      )
    )

    R1 <- if (all(is.finite(lin_pred))) {
      distance_correlation(lin_pred, y_test)
    } else {
      NA_real_
    }

    ace_fit <- fit_ace_safely(
      x_train,
      y_train,
      span
    )

    ace_test <- apply_fitted_ace_to_test(
      ace_fit,
      x_train,
      y_train,
      x_test,
      y_test
    )

    R2 <- ace_test$R2
    R3 <- ace_test$R3

    inactive_ratio <- if (ace_fit$success) {
      transformation_variance_ratio(
        ace_fit$tx,
        active_s
      )
    } else {
      NA_real_
    }

    rows[[ii]] <- data.frame(
      fold = fold,
      n_train = length(train_index),
      n_test = length(test_index),
      R0 = R0,
      R0_raw = R0_raw,
      R1 = R1,
      R2 = R2,
      R3 = R3,
      GP = R2 - R1,
      GR = R3 - R2,
      delta10 = R1 - R0,
      delta20 = R2 - R0,
      delta30 = R3 - R0,
      inactive_tx_variance_ratio = inactive_ratio,
      ace_success = ace_test$success,
      ace_error_message =
        if (is.null(ace_test$error_message)) {
          NA_character_
        } else {
          ace_test$error_message
        },
      stringsAsFactors = FALSE
    )
  }

  tab <- do.call(rbind, rows)

  list(
    success = all(tab$ace_success),
    success_rate = mean(tab$ace_success),
    R0 = safe_mean(tab$R0),
    R0_raw = safe_mean(tab$R0_raw),
    R1 = safe_mean(tab$R1),
    R2 = safe_mean(tab$R2),
    R3 = safe_mean(tab$R3),
    GP = safe_mean(tab$GP),
    GR = safe_mean(tab$GR),
    delta10 = safe_mean(tab$delta10),
    delta20 = safe_mean(tab$delta20),
    delta30 = safe_mean(tab$delta30),
    inactive_tx_variance_ratio =
      safe_mean(tab$inactive_tx_variance_ratio),
    error_message = {
      ee <- unique(
        tab$ace_error_message[
          !is.na(tab$ace_error_message) &
            nzchar(tab$ace_error_message)
        ]
      )

      if (length(ee) == 0L) {
        NA_character_
      } else {
        paste(ee, collapse = " | ")
      }
    },
    fold_table = tab,
    fold_id = fold_id
  )
}


# -----------------------------------------------------------------------------
# 9. Deterministic seeds and one Monte Carlo replication
# -----------------------------------------------------------------------------
extended_replication_seed <- function(
    condition,
    n,
    p,
    s,
    replication,
    target_r2
) {
  ci <- match(condition, extended_condition_levels)

  if (is.na(ci)) {
    stop("Unknown condition in seed function: ", condition)
  }

  r2code <- as.integer(round(target_r2 * 100))

  seed_num <-
    BASE_SEED +
    ci * 100000000 +
    n * 10000 +
    p * 1000 +
    s * 100 +
    r2code * 10 +
    replication

  # Keep seed inside R's valid integer range.
  as.integer((seed_num %% 2147483646) + 1)
}

run_extended_replication <- function(
    condition,
    n,
    p,
    s,
    replication,
    target_r2,
    span = ACE_SPAN,
    k = CF_FOLDS
) {
  seed <- extended_replication_seed(
    condition = condition,
    n = n,
    p = p,
    s = s,
    replication = replication,
    target_r2 = target_r2
  )

  t0 <- proc.time()[3]

  dat <- generate_extended_data(
    condition = condition,
    n = n,
    p = p,
    s = s,
    target_r2 = target_r2,
    seed = seed
  )

  ins <- insample_metrics(
    dat$x,
    dat$y,
    active_s = s,
    span = span,
    standardize_x = SIM_STANDARDIZE_X
  )

  cf <- crossfit_metrics(
    dat$x,
    dat$y,
    active_s = s,
    k = k,
    span = span,
    seed = seed + 500000L,
    standardize_x = SIM_STANDARDIZE_X
  )

  elapsed <- proc.time()[3] - t0

  data.frame(
    condition = condition,
    condition_id = unname(extended_condition_ids[condition]),
    n = n,
    p = p,
    s = s,
    n_inactive = p - s,
    target_r2 = target_r2,
    replication = replication,
    seed = seed,
    sample_eta_mean = dat$sample_eta_mean,
    sample_eta_var = dat$sample_eta_var,
    sample_epsilon_var = dat$sample_epsilon_var,
    sample_y_var = dat$sample_y_var,
    sample_signal_fraction = dat$sample_signal_fraction,
    R0 = ins$R0,
    R0_raw = ins$R0_raw,
    R1 = ins$R1,
    R2 = ins$R2,
    R3 = ins$R3,
    GP = ins$GP,
    GR = ins$GR,
    delta10 = ins$delta10,
    delta20 = ins$delta20,
    delta30 = ins$delta30,
    cf_R0 = cf$R0,
    cf_R0_raw = cf$R0_raw,
    cf_R1 = cf$R1,
    cf_R2 = cf$R2,
    cf_R3 = cf$R3,
    cf_GP = cf$GP,
    cf_GR = cf$GR,
    cf_delta10 = cf$delta10,
    cf_delta20 = cf$delta20,
    cf_delta30 = cf$delta30,
    ace_rsq = ins$ace_rsq,
    inactive_tx_variance_ratio =
      ins$inactive_tx_variance_ratio,
    cf_inactive_tx_variance_ratio =
      cf$inactive_tx_variance_ratio,
    ace_success = ins$ace_success,
    cf_success = cf$success,
    cf_success_rate = cf$success_rate,
    ace_error_message =
      if (is.null(ins$ace_error_message)) {
        NA_character_
      } else {
        ins$ace_error_message
      },
    cf_error_message =
      if (is.null(cf$error_message)) {
        NA_character_
      } else {
        cf$error_message
      },
    runtime_seconds = as.numeric(elapsed),
    stringsAsFactors = FALSE
  )
}


# -----------------------------------------------------------------------------
# 10. Preflight tests
# -----------------------------------------------------------------------------

ace_startup_selftest <- function() {
  ctrl <- make_ace_control(
    ACE_SPAN,
    ACE_MAXIT
  )

  if (!identical(typeof(ctrl$span), "double") ||
      !identical(typeof(ctrl$maxit), "double")) {
    stop(
      "ACE preflight failed: span and maxit must both be double."
    )
  }

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
        "ACE preflight failed in acepack::set_control(). ",
        "acepack version=", ACEPACK_VERSION,
        "; error=", conditionMessage(ctrl_test)
      )
    )
  }

  set.seed(BASE_SEED + 4242L)

  n_test <- 120L
  x_test <- cbind(
    X1 = stats::rnorm(n_test),
    X2 = stats::rnorm(n_test)
  )

  y_test <-
    x_test[, 1]^2 +
    0.5 * x_test[, 2] +
    stats::rnorm(n_test, sd = 0.5)

  ans <- fit_ace_safely(
    x_test,
    y_test,
    span = ACE_SPAN
  )

  if (!isTRUE(ans$success)) {
    stop(
      paste0(
        "ACE startup self-test FAILED. ",
        "acepack version=", ACEPACK_VERSION,
        "; error=", ans$error_message
      )
    )
  }

  if (nrow(ans$tx) != n_test ||
      ncol(ans$tx) != 2L ||
      !all(is.finite(ans$tx)) ||
      !all(is.finite(ans$ty)) ||
      !is.finite(ans$fit$rsq)) {
    stop(
      "ACE startup self-test returned malformed/non-finite output."
    )
  }

  log_message(
    "[ACE SELF-TEST PASS] acepack=", ACEPACK_VERSION,
    "; span=", ACE_SPAN,
    "; maxit=", ACE_MAXIT,
    "; rsq=", sprintf("%.4f", ans$fit$rsq),
    "; elapsed=", sprintf("%.3fs", ans$elapsed_seconds)
  )

  invisible(TRUE)
}


signal_moment_preflight <- function(n_moment = 20000L) {
  rows <- list()
  ix <- 1L

  for (cond in extended_condition_levels) {
    for (s0 in EXT_S_VALUES) {
      p0 <- max(5L, s0)

      set.seed(
        BASE_SEED +
          match(cond, extended_condition_levels) * 10000L +
          s0
      )

      x <- make_x_matrix_extended(
        n = n_moment,
        p = p0
      )

      eta <- raw_extended_signal(
        x,
        condition = cond,
        s = s0
      )

      m <- mean(eta)
      v <- stats::var(eta)

      rows[[ix]] <- data.frame(
        condition = cond,
        condition_id =
          unname(extended_condition_ids[cond]),
        p = p0,
        s = s0,
        n_moment = n_moment,
        eta_mean = m,
        eta_variance = v,
        all_finite = all(is.finite(eta)),
        stringsAsFactors = FALSE
      )

      ix <- ix + 1L
    }
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL

  write.csv(
    out,
    file.path(
      ROOT_OUTPUT,
      "PREFLIGHT_signal_moments_P6_P12.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  ok <-
    all(out$all_finite) &&
    all(is.finite(out$eta_variance)) &&
    all(out$eta_variance > 0.5) &&
    all(out$eta_variance < 1.5) &&
    all(abs(out$eta_mean) < 0.15)

  if (!ok) {
    stop(
      paste0(
        "Signal-moment preflight FAILED. ",
        "Inspect PREFLIGHT_signal_moments_P6_P12.csv. ",
        "The long simulation has NOT started."
      )
    )
  }

  log_message(
    "[SIGNAL MOMENT PREFLIGHT PASS] ",
    nrow(out),
    " P6--P12 x s checks; empirical eta variance remained near 1."
  )

  invisible(out)
}


pipeline_smoke_preflight <- function() {
  rows <- list()
  ix <- 1L

  log_message(
    "[PIPELINE PREFLIGHT] Starting ",
    length(extended_condition_levels) * length(EXT_S_VALUES),
    " P6--P12 x s smoke checks."
  )

  for (cond in extended_condition_levels) {
    for (s0 in EXT_S_VALUES) {
      p0 <- max(5L, s0)

      ans <- tryCatch(
        run_extended_replication(
          condition = cond,
          n = 200L,
          p = p0,
          s = s0,
          replication = 1L,
          target_r2 = 0.5,
          span = ACE_SPAN,
          k = CF_FOLDS
        ),
        error = function(e) e
      )

      if (inherits(ans, "error")) {
        rows[[ix]] <- data.frame(
          condition = cond,
          condition_id =
            unname(extended_condition_ids[cond]),
          p = p0,
          s = s0,
          sample_eta_var = NA_real_,
          ace_success = FALSE,
          cf_success = FALSE,
          finite_R0_R3 = FALSE,
          finite_cf_R0_R3 = FALSE,
          error_message = conditionMessage(ans),
          stringsAsFactors = FALSE
        )
      } else {
        finite_in <- all(
          is.finite(
            unlist(
              ans[
                1,
                c("R0", "R1", "R2", "R3", "GP", "GR"),
                drop = FALSE
              ]
            )
          )
        )

        finite_cf <- all(
          is.finite(
            unlist(
              ans[
                1,
                c(
                  "cf_R0", "cf_R1", "cf_R2",
                  "cf_R3", "cf_GP", "cf_GR"
                ),
                drop = FALSE
              ]
            )
          )
        )

        err_txt <- c(
          ans$ace_error_message[1],
          ans$cf_error_message[1]
        )
        err_txt <- err_txt[
          !is.na(err_txt) &
            nzchar(err_txt)
        ]

        rows[[ix]] <- data.frame(
          condition = cond,
          condition_id =
            unname(extended_condition_ids[cond]),
          p = p0,
          s = s0,
          sample_eta_var = ans$sample_eta_var[1],
          ace_success = isTRUE(ans$ace_success[1]),
          cf_success = isTRUE(ans$cf_success[1]),
          finite_R0_R3 = finite_in,
          finite_cf_R0_R3 = finite_cf,
          error_message =
            if (length(err_txt) == 0L) {
              NA_character_
            } else {
              paste(err_txt, collapse = " | ")
            },
          stringsAsFactors = FALSE
        )
      }

      ix <- ix + 1L
    }
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL

  write.csv(
    out,
    file.path(
      ROOT_OUTPUT,
      "PREFLIGHT_pipeline_smoke_P6_P12.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  ok <-
    all(is.finite(out$sample_eta_var)) &&
    all(out$ace_success) &&
    all(out$cf_success) &&
    all(out$finite_R0_R3) &&
    all(out$finite_cf_R0_R3)

  if (!ok) {
    stop(
      paste0(
        "Extended pipeline smoke preflight FAILED. ",
        "Inspect PREFLIGHT_pipeline_smoke_P6_P12.csv. ",
        "The 504,000-replication simulation has NOT started."
      )
    )
  }

  log_message(
    "[PIPELINE SELF-TEST PASS] all ",
    nrow(out),
    " P6--P12 x s smoke checks returned finite ",
    "R0--R3, GP/GR, and successful ACE/cross-fitting."
  )

  invisible(out)
}


# -----------------------------------------------------------------------------
# 11. Progress / parallel helper
# -----------------------------------------------------------------------------
format_duration <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("NA")

  seconds <- round(seconds)
  d <- seconds %/% 86400L
  h <- (seconds %% 86400L) %/% 3600L
  m <- (seconds %% 3600L) %/% 60L
  sec <- seconds %% 60L

  if (d > 0L) {
    return(sprintf("%dd %02dh %02dm", d, h, m))
  }

  if (h > 0L) {
    return(sprintf("%02dh %02dm %02ds", h, m, sec))
  }

  if (m > 0L) {
    return(sprintf("%02dm %02ds", m, sec))
  }

  sprintf("%02ds", sec)
}

format_eta_clock <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("NA")
  format(
    Sys.time() + seconds,
    "%Y-%m-%d %H:%M:%S"
  )
}

parallel_rows <- function(
    grid,
    fun,
    label = "tasks",
    batch_size = PROGRESS_BATCH_MAIN,
    overall_completed_before = 0L,
    overall_total = NULL,
    overall_label = NULL
) {
  if (nrow(grid) == 0L) return(data.frame())

  total <- nrow(grid)
  batch_size <- max(
    1L,
    min(as.integer(batch_size), total)
  )

  ids <- seq_len(total)
  batch_id <- ceiling(ids / batch_size)
  batches <- split(ids, batch_id)
  parts <- vector("list", length(batches))

  t0 <- Sys.time()
  completed <- 0L

  log_message(
    "Starting ", label,
    ": ", total,
    " tasks; workers=", MAX_WORKERS,
    "; progress batch=", batch_size
  )

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
      elapsed <- as.numeric(
        difftime(
          Sys.time(),
          t0,
          units = "secs"
        )
      )

      pct <- 100 * completed / total

      rate_per_min <- if (elapsed > 0) {
        60 * completed / elapsed
      } else {
        NA_real_
      }

      eta_sec <- if (completed > 0 && elapsed > 0) {
        elapsed * (total - completed) / completed
      } else {
        NA_real_
      }

      msg <- sprintf(
        "[%s] %d/%d (%.1f%%) | elapsed %s | ETA %s | finish ~ %s | %.2f tasks/min",
        label,
        completed,
        total,
        pct,
        format_duration(elapsed),
        format_duration(eta_sec),
        format_eta_clock(eta_sec),
        rate_per_min
      )

      if (!is.null(overall_total) &&
          is.finite(overall_total) &&
          overall_total > 0) {
        overall_done <- min(
          overall_total,
          overall_completed_before + completed
        )

        overall_pct <-
          100 * overall_done / overall_total

        scope <- if (is.null(overall_label)) {
          "overall"
        } else {
          overall_label
        }

        msg <- paste0(
          msg,
          sprintf(
            " | %s %d/%d (%.1f%% tasks)",
            scope,
            overall_done,
            overall_total,
            overall_pct
          )
        )
      }

      log_message(msg)
    }
  }

  out <- do.call(rbind, parts)
  rownames(out) <- NULL

  elapsed_total <- as.numeric(
    difftime(
      Sys.time(),
      t0,
      units = "secs"
    )
  )

  log_message(
    "Completed ", label,
    ": ", nrow(out),
    " rows; elapsed=", format_duration(elapsed_total)
  )

  out
}


# -----------------------------------------------------------------------------
# 12. Monte Carlo summary
# -----------------------------------------------------------------------------
summarize_extended_mc <- function(results) {
  group_vars <- c(
    "condition",
    "condition_id",
    "n",
    "p",
    "s",
    "n_inactive",
    "target_r2"
  )

  metrics <- c(
    "sample_eta_mean",
    "sample_eta_var",
    "sample_epsilon_var",
    "sample_y_var",
    "sample_signal_fraction",
    "R0", "R1", "R2", "R3", "GP", "GR",
    "cf_R0", "cf_R1", "cf_R2", "cf_R3",
    "cf_GP", "cf_GR",
    "ace_rsq",
    "inactive_tx_variance_ratio",
    "cf_inactive_tx_variance_ratio",
    "runtime_seconds"
  )

  keys <- unique(
    results[, group_vars, drop = FALSE]
  )

  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    key <- keys[i, , drop = FALSE]
    keep <- rep(TRUE, nrow(results))

    for (v in group_vars) {
      keep <- keep &
        (results[[v]] == key[[v]][1])
    }

    sub <- results[keep, , drop = FALSE]
    row <- key

    row$replications <- nrow(sub)
    row$ace_success_rate <-
      mean(sub$ace_success, na.rm = TRUE)
    row$cf_success_rate <-
      mean(sub$cf_success, na.rm = TRUE)
    row$finite_cf_GP_rate <-
      mean(is.finite(sub$cf_GP))
    row$finite_cf_GR_rate <-
      mean(is.finite(sub$cf_GR))
    row$cf_GP_positive_rate <-
      safe_mean(as.numeric(sub$cf_GP > 0))
    row$cf_GR_positive_rate <-
      safe_mean(as.numeric(sub$cf_GR > 0))

    for (m in metrics) {
      x <- sub[[m]]
      mn <- safe_mean(x)
      se <- safe_se(x)

      row[[paste0(m, "_mean")]] <- mn
      row[[paste0(m, "_sd")]] <- safe_sd(x)
      row[[paste0(m, "_se")]] <- se
      row[[paste0(m, "_q025")]] <-
        safe_quantile(x, 0.025)
      row[[paste0(m, "_median")]] <-
        safe_quantile(x, 0.5)
      row[[paste0(m, "_q975")]] <-
        safe_quantile(x, 0.975)
      row[[paste0(m, "_mc95_lower")]] <-
        if (is.finite(mn) && is.finite(se)) {
          mn - 1.96 * se
        } else {
          NA_real_
        }
      row[[paste0(m, "_mc95_upper")]] <-
        if (is.finite(mn) && is.finite(se)) {
          mn + 1.96 * se
        } else {
          NA_real_
        }
    }

    rows[[i]] <- row
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}


# -----------------------------------------------------------------------------
# 13. Extended main simulation: 504,000 replications
# -----------------------------------------------------------------------------
run_extended_main <- function() {
  log_message("============================================================")
  log_message("[Track F] Extended P6--P12 main simulation started.")
  log_message(
    "[Track F] Total planned replications = ",
    format(EXPECTED_REPLICATIONS, big.mark = ",")
  )
  log_message("============================================================")

  all_blocks <- list()
  block_ix <- 1L
  overall_completed <- 0L

  total_blocks <-
    length(EXT_N_VALUES) *
    nrow(valid_ps) *
    length(extended_condition_levels)

  current_block <- 0L

  for (n0 in EXT_N_VALUES) {
    for (ps_ix in seq_len(nrow(valid_ps))) {
      p0 <- as.integer(valid_ps$p[ps_ix])
      s0 <- as.integer(valid_ps$s[ps_ix])

      for (cond in extended_condition_levels) {
        current_block <- current_block + 1L

        cond_id <- unname(
          extended_condition_ids[cond]
        )

        checkpoint <- file.path(
          CHECKPOINT_OUTPUT,
          paste0(
            "trackF_n", n0,
            "_p", p0,
            "_s", s0,
            "_", cond_id,
            ".rds"
          )
        )

        expected_rows <-
          length(EXT_R2_VALUES) * EXT_B

        if (RESUME_FROM_CHECKPOINTS &&
            file.exists(checkpoint)) {
          tmp <- tryCatch(
            readRDS(checkpoint),
            error = function(e) NULL
          )

          if (!is.null(tmp) &&
              nrow(tmp) == expected_rows) {
            overall_completed <-
              overall_completed + expected_rows

            log_message(
              "[Track F] Reusing checkpoint ",
              current_block, "/", total_blocks,
              " | n=", n0,
              ", p=", p0,
              ", s=", s0,
              ", ", cond_id,
              " | overall ",
              format(overall_completed, big.mark = ","),
              "/",
              format(EXPECTED_REPLICATIONS, big.mark = ","),
              " (",
              sprintf(
                "%.1f",
                100 * overall_completed /
                  EXPECTED_REPLICATIONS
              ),
              "%)"
            )

            all_blocks[[block_ix]] <- tmp
            block_ix <- block_ix + 1L
            next
          }
        }

        g <- expand.grid(
          target_r2 = EXT_R2_VALUES,
          replication = seq_len(EXT_B),
          KEEP.OUT.ATTRS = FALSE,
          stringsAsFactors = FALSE
        )

        label <- paste0(
          "Track F block ",
          current_block, "/", total_blocks,
          " n=", n0,
          ", p=", p0,
          ", s=", s0,
          ", ", cond_id
        )

        res <- parallel_rows(
          g,
          function(z) {
            run_extended_replication(
              condition = cond,
              n = n0,
              p = p0,
              s = s0,
              replication =
                as.integer(z$replication[1]),
              target_r2 =
                as.numeric(z$target_r2[1]),
              span = ACE_SPAN,
              k = CF_FOLDS
            )
          },
          label = label,
          batch_size = PROGRESS_BATCH_MAIN,
          overall_completed_before =
            overall_completed,
          overall_total =
            EXPECTED_REPLICATIONS,
          overall_label =
            "Track F overall"
        )

        saveRDS(res, checkpoint)

        overall_completed <-
          overall_completed + nrow(res)

        all_blocks[[block_ix]] <- res
        block_ix <- block_ix + 1L
      }
    }
  }

  results <- do.call(rbind, all_blocks)
  rownames(results) <- NULL

  # Save raw results BEFORE summaries/plots.
  raw_rds <- file.path(
    SIM_OUTPUT,
    "F_extended_P6_P12_replications.rds"
  )

  raw_csv <- file.path(
    SIM_OUTPUT,
    "F_extended_P6_P12_replications.csv"
  )

  saveRDS(results, raw_rds)

  write.csv(
    results,
    raw_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  log_message(
    "[Track F] Raw combined results saved: ",
    normalizePath(
      raw_rds,
      winslash = "/",
      mustWork = FALSE
    )
  )

  # Overall computational integrity.
  integrity <- data.frame(
    n_rows = nrow(results),
    expected_rows = EXPECTED_REPLICATIONS,
    ace_success_rate =
      safe_mean(as.numeric(results$ace_success)),
    cf_success_rate =
      safe_mean(as.numeric(results$cf_success)),
    finite_cf_R0_rate =
      mean(is.finite(results$cf_R0)),
    finite_cf_R1_rate =
      mean(is.finite(results$cf_R1)),
    finite_cf_R2_rate =
      mean(is.finite(results$cf_R2)),
    finite_cf_R3_rate =
      mean(is.finite(results$cf_R3)),
    finite_cf_GP_rate =
      mean(is.finite(results$cf_GP)),
    finite_cf_GR_rate =
      mean(is.finite(results$cf_GR)),
    stringsAsFactors = FALSE
  )

  write.csv(
    integrity,
    file.path(
      SIM_OUTPUT,
      "F_extended_integrity_check.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  log_message(
    "[Track F] Integrity: rows=",
    integrity$n_rows,
    "/",
    integrity$expected_rows,
    "; ACE success=",
    sprintf("%.4f", integrity$ace_success_rate),
    "; CF success=",
    sprintf("%.4f", integrity$cf_success_rate),
    "; finite GP=",
    sprintf("%.4f", integrity$finite_cf_GP_rate),
    "; finite GR=",
    sprintf("%.4f", integrity$finite_cf_GR_rate)
  )

  # Save failure frequencies if present.
  for (err_col in c(
    "ace_error_message",
    "cf_error_message"
  )) {
    ee <- results[[err_col]]
    ee <- ee[
      !is.na(ee) &
        nzchar(ee)
    ]

    if (length(ee) > 0L) {
      tab <- as.data.frame(
        sort(
          table(ee),
          decreasing = TRUE
        ),
        stringsAsFactors = FALSE
      )

      names(tab) <- c(
        "error_message",
        "count"
      )

      write.csv(
        tab,
        file.path(
          SIM_OUTPUT,
          paste0(
            "F_",
            err_col,
            "_frequencies.csv"
          )
        ),
        row.names = FALSE,
        fileEncoding = "UTF-8"
      )
    }
  }

  # Cell-level summaries.
  summary <- summarize_extended_mc(results)

  write.csv(
    summary,
    file.path(
      SIM_OUTPUT,
      "F_extended_P6_P12_summary.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  # Flag cells that need manual inspection.
  qc_cells <- summary[
    summary$ace_success_rate < 0.95 |
      summary$cf_success_rate < 0.95 |
      summary$finite_cf_GP_rate < 0.95 |
      summary$finite_cf_GR_rate < 0.95,
    c(
      "condition",
      "condition_id",
      "n",
      "p",
      "s",
      "target_r2",
      "ace_success_rate",
      "cf_success_rate",
      "finite_cf_GP_rate",
      "finite_cf_GR_rate"
    ),
    drop = FALSE
  ]

  write.csv(
    qc_cells,
    file.path(
      SIM_OUTPUT,
      "F_cells_requiring_QC_review.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  if (nrow(qc_cells) > 0L) {
    log_message(
      "[Track F WARNING] ",
      nrow(qc_cells),
      " cells have success/finite rates below 0.95. ",
      "Inspect F_cells_requiring_QC_review.csv."
    )
  } else {
    log_message(
      "[Track F QC PASS] All cells have ",
      "ACE/CF/finite GP/GR rates >= 0.95."
    )
  }

  # Representative publication table.
  representative <- summary[
    summary$n == 500L &
      abs(summary$target_r2 - 0.5) <
        .Machine$double.eps^0.5,
    ,
    drop = FALSE
  ]

  representative <- representative[
    order(
      representative$p,
      representative$s,
      representative$condition_id
    ),
    ,
    drop = FALSE
  ]

  keep_rep <- c(
    "condition",
    "condition_id",
    "n",
    "p",
    "s",
    "n_inactive",
    "target_r2",
    "replications",
    "cf_R0_mean",
    "cf_R1_mean",
    "cf_R2_mean",
    "cf_R3_mean",
    "cf_GP_mean",
    "cf_GP_sd",
    "cf_GP_q025",
    "cf_GP_median",
    "cf_GP_q975",
    "cf_GP_positive_rate",
    "cf_GR_mean",
    "cf_GR_sd",
    "cf_GR_q025",
    "cf_GR_median",
    "cf_GR_q975",
    "cf_GR_positive_rate",
    "cf_inactive_tx_variance_ratio_mean",
    "ace_success_rate",
    "cf_success_rate"
  )

  write.csv(
    representative[, keep_rep, drop = FALSE],
    file.path(
      PAPER_OUTPUT,
      "F_representative_n500_R2_05.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  # Signal/SNR diagnostics.
  signal_diag <- summary[
    ,
    c(
      "condition",
      "condition_id",
      "n",
      "p",
      "s",
      "target_r2",
      "sample_eta_mean_mean",
      "sample_eta_var_mean",
      "sample_eta_var_sd",
      "sample_signal_fraction_mean",
      "sample_signal_fraction_sd"
    ),
    drop = FALSE
  ]

  write.csv(
    signal_diag,
    file.path(
      PAPER_OUTPUT,
      "F_signal_variance_and_realized_SNR.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  # Active-dimension view: useful for separating total p from active s.
  active_view <- summary[
    summary$n == 500L &
      abs(summary$target_r2 - 0.5) <
        .Machine$double.eps^0.5,
    c(
      "condition",
      "condition_id",
      "p",
      "s",
      "n_inactive",
      "cf_GP_mean",
      "cf_GR_mean",
      "cf_R1_mean",
      "cf_R2_mean",
      "cf_R3_mean",
      "cf_inactive_tx_variance_ratio_mean"
    ),
    drop = FALSE
  ]

  write.csv(
    active_view,
    file.path(
      PAPER_OUTPUT,
      "F_active_dimension_effect_n500_R2_05.csv"
    ),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  log_message("[Track F] Extended main simulation completed.")
  list(
    results = results,
    summary = summary,
    integrity = integrity
  )
}


# -----------------------------------------------------------------------------
# 14. Figures for paper/supplement
# -----------------------------------------------------------------------------
make_extended_figures <- function(summary) {
  # GP and GR vs active dimension s.
  pdf(
    file.path(
      PAPER_OUTPUT,
      "F_GP_GR_by_active_s.pdf"
    ),
    width = 9,
    height = 6.5,
    onefile = TRUE
  )

  for (n0 in EXT_N_VALUES) {
    for (p0 in EXT_P_VALUES) {
      valid_s0 <- valid_ps$s[
        valid_ps$p == p0
      ]

      for (r20 in EXT_R2_VALUES) {
        d0 <- summary[
          summary$n == n0 &
            summary$p == p0 &
            summary$s %in% valid_s0 &
            abs(summary$target_r2 - r20) <
              .Machine$double.eps^0.5,
          ,
          drop = FALSE
        ]

        for (metric in c(
          "cf_GP_mean",
          "cf_GR_mean"
        )) {
          vals <- d0[[metric]]
          vals <- vals[is.finite(vals)]

          yr <- if (length(vals) > 0L) {
            range(c(0, vals), finite = TRUE)
          } else {
            c(-0.01, 0.01)
          }

          if (diff(yr) == 0) {
            yr <- yr + c(-0.01, 0.01)
          }

          plot(
            NA,
            xlim = range(valid_s0),
            ylim = yr,
            xlab = "Number of active predictors s",
            ylab = metric,
            main = paste0(
              metric,
              " | n=", n0,
              ", p=", p0,
              ", target R^2=", r20
            ),
            xaxt = "n"
          )

          axis(
            1,
            at = valid_s0,
            labels = valid_s0
          )

          grid(col = "grey90", lty = 1)
          abline(h = 0, lty = 2, col = "grey50")

          for (cond in extended_condition_levels) {
            d <- d0[
              d0$condition == cond,
              ,
              drop = FALSE
            ]

            d <- d[
              order(d$s),
              ,
              drop = FALSE
            ]

            ok <- is.finite(d[[metric]])

            if (any(ok)) {
              lines(
                d$s[ok],
                d[[metric]][ok],
                type = "b",
                pch = scenario_point_shapes[cond],
                lty = scenario_line_types[cond],
                lwd = 2,
                col = scenario_colors[cond]
              )
            }
          }

          legend(
            "topright",
            legend = extended_condition_ids[extended_condition_levels],
            col = scenario_colors[extended_condition_levels],
            pch = scenario_point_shapes[extended_condition_levels],
            lty = scenario_line_types[extended_condition_levels],
            lwd = 2,
            bty = "n",
            cex = 0.75,
            pt.cex = 1
          )
        }
      }
    }
  }

  dev.off()

  # Representative GP-GR maps at n=500, R2=.5, one page per total p.
  pdf(
    file.path(
      PAPER_OUTPUT,
      "F_GP_GR_maps_n500_R2_05.pdf"
    ),
    width = 8.5,
    height = 7,
    onefile = TRUE
  )

  for (p0 in EXT_P_VALUES) {
    d <- summary[
      summary$n == 500L &
        summary$p == p0 &
        abs(summary$target_r2 - 0.5) <
          .Machine$double.eps^0.5 &
        is.finite(summary$cf_GP_mean) &
        is.finite(summary$cf_GR_mean),
      ,
      drop = FALSE
    ]

    if (nrow(d) == 0L) {
      plot.new()
      title(
        main = paste0(
          "GP-GR map | p=", p0
        )
      )
      text(
        0.5,
        0.5,
        "No finite values"
      )
      next
    }

    xr <- range(
      c(0, d$cf_GP_mean),
      finite = TRUE
    )

    yr <- range(
      c(0, d$cf_GR_mean),
      finite = TRUE
    )

    if (diff(xr) == 0) {
      xr <- xr + c(-0.01, 0.01)
    }

    if (diff(yr) == 0) {
      yr <- yr + c(-0.01, 0.01)
    }

    plot(
      d$cf_GP_mean,
      d$cf_GR_mean,
      pch = scenario_point_shapes[d$condition],
      col = scenario_colors[d$condition],
      bg = scenario_colors[d$condition],
      cex = 1.2,
      xlim = xr,
      ylim = yr,
      xlab = "Cross-fitted GP = R2 - R1",
      ylab = "Cross-fitted GR = R3 - R2",
      main = paste0(
        "Extended diagnostic map | n=500, p=",
        p0,
        ", R^2=0.5"
      )
    )

    grid(col = "grey90", lty = 1)
    abline(
      h = 0,
      v = 0,
      lty = 2,
      col = "grey50"
    )

    for (cond in extended_condition_levels) {
      dd <- d[d$condition == cond, , drop = FALSE]
      if (nrow(dd) > 1L) {
        ord <- order(dd$s)
        lines(
          dd$cf_GP_mean[ord],
          dd$cf_GR_mean[ord],
          col = scenario_colors[cond],
          lty = scenario_line_types[cond],
          lwd = 1.7
        )
      }
    }

    text(
      d$cf_GP_mean,
      d$cf_GR_mean,
      labels = paste0(
        d$condition_id,
        "-s", d$s
      ),
      pos = 3,
      cex = 0.72,
      col = scenario_colors[d$condition]
    )

    legend(
      "topright",
      legend = extended_condition_ids[extended_condition_levels],
      col = scenario_colors[extended_condition_levels],
      pch = scenario_point_shapes[extended_condition_levels],
      lty = scenario_line_types[extended_condition_levels],
      lwd = 2,
      bty = "n",
      cex = 0.8,
      pt.cex = 1
    )
  }

  dev.off()

  log_message(
    "[Figures] Extended GP/GR figures saved in ",
    normalizePath(
      PAPER_OUTPUT,
      winslash = "/",
      mustWork = FALSE
    )
  )

  invisible(TRUE)
}


# -----------------------------------------------------------------------------
# 15. Save configuration/session information
# -----------------------------------------------------------------------------
config_lines <- c(
  paste0("timestamp=", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste0("PROJECT_DIR=", PROJECT_DIR),
  paste0("ROOT_OUTPUT=", ROOT_OUTPUT),
  paste0("R_version=", R.version.string),
  paste0("acepack_version=", ACEPACK_VERSION),
  paste0("workers=", MAX_WORKERS),
  paste0("ACE_SPAN=", ACE_SPAN),
  paste0("ACE_MAXIT=", ACE_MAXIT),
  paste0("CF_FOLDS=", CF_FOLDS),
  paste0("n=", paste(EXT_N_VALUES, collapse = ",")),
  paste0("p=", paste(EXT_P_VALUES, collapse = ",")),
  paste0("s=", paste(EXT_S_VALUES, collapse = ","), "; s<=p"),
  paste0("R2=", paste(EXT_R2_VALUES, collapse = ",")),
  paste0("B=", EXT_B),
  paste0("conditions=", paste(extended_condition_ids, collapse = ",")),
  paste0("expected_replications=", EXPECTED_REPLICATIONS),
  "error_distribution=normal",
  "predictor_distribution=iid standard normal",
  "simulation_X_standardization=FALSE",
  "bootstrap_on_full_grid=FALSE",
  "figure_output=colored base-R publication-style plots"
)

writeLines(
  config_lines,
  file.path(
    ROOT_OUTPUT,
    "analysis_configuration_extended.txt"
  ),
  useBytes = TRUE
)

writeLines(
  capture.output(sessionInfo()),
  file.path(
    ROOT_OUTPUT,
    "sessionInfo_preflight.txt"
  ),
  useBytes = TRUE
)


# -----------------------------------------------------------------------------
# 16. Mandatory preflight
# -----------------------------------------------------------------------------
log_message("============================================================")
log_message("MANDATORY PREFLIGHT START")
log_message("============================================================")

ace_startup_selftest()
signal_moment_preflight(n_moment = 20000L)
pipeline_smoke_preflight()

writeLines(
  c(
    paste0(
      "timestamp=",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    ),
    paste0(
      "acepack_version=",
      ACEPACK_VERSION
    ),
    "ACE_SELF_TEST=PASS",
    "SIGNAL_MOMENT_PREFLIGHT=PASS",
    "PIPELINE_SMOKE_P6_P12_X_S=PASS",
    "LONG_SIMULATION_ALLOWED=TRUE"
  ),
  file.path(
    ROOT_OUTPUT,
    "PREFLIGHT_PASS.txt"
  ),
  useBytes = TRUE
)

log_message("============================================================")
log_message("ALL PREFLIGHT CHECKS PASSED")
log_message("The 504,000-replication main simulation is safe to start.")
log_message("============================================================")

if (STOP_AFTER_PREFLIGHT) {
  log_message(
    "STOP_AFTER_PREFLIGHT=TRUE: stopping before long simulation."
  )

  if (MAX_WORKERS > 1L) {
    future::plan(future::sequential)
  }

  stop(
    "Preflight completed successfully. ",
    "Set STOP_AFTER_PREFLIGHT <- FALSE to run the main simulation."
  )
}


# -----------------------------------------------------------------------------
# 17. Execute Extended Main Simulation
# -----------------------------------------------------------------------------
t_run_start <- Sys.time()

trackF <- run_extended_main()

make_extended_figures(
  trackF$summary
)

elapsed_all <- as.numeric(
  difftime(
    Sys.time(),
    t_run_start,
    units = "secs"
  )
)

writeLines(
  c(
    paste0(
      "completed_at=",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    ),
    paste0(
      "total_elapsed=",
      format_duration(elapsed_all)
    ),
    paste0(
      "replications=",
      EXPECTED_REPLICATIONS
    ),
    paste0(
      "raw_results=",
      file.path(
        SIM_OUTPUT,
        "F_extended_P6_P12_replications.rds"
      )
    ),
    paste0(
      "summary=",
      file.path(
        SIM_OUTPUT,
        "F_extended_P6_P12_summary.csv"
      )
    )
  ),
  file.path(
    ROOT_OUTPUT,
    "RUN_COMPLETED.txt"
  ),
  useBytes = TRUE
)

log_message("============================================================")
log_message("EXTENDED P6--P12 SIMULATION COMPLETED")
log_message(
  "Total elapsed: ",
  format_duration(elapsed_all)
)
log_message(
  "Results: ",
  normalizePath(
    ROOT_OUTPUT,
    winslash = "/",
    mustWork = FALSE
  )
)
log_message("============================================================")

# Return to sequential plan so RStudio closes workers cleanly.
if (MAX_WORKERS > 1L) {
  future::plan(future::sequential)
}
