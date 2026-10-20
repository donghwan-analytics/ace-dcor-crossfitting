# Checks only: no analysis execution or package installation.
scripts <- list.files("R", pattern = "[.]R$", full.names = TRUE)
if (!length(scripts)) stop("Run this script from the repository root.")
for (script in scripts) {
  parse(file = script, encoding = "UTF-8")
  message("Syntax OK: ", script)
}
packages <- readLines("requirements-r.txt", warn = FALSE)
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "))
message("Basic checks passed. Verify data, paths and analysis settings separately.")
