# Deterministic reference runs (no random restarts) of the CRAN msPCA package.
# Inputs/outputs are exchanged as raw little-endian float64 so the comparison is bit-exact.
suppressMessages(library(msPCA))
rd <- function(f) { con <- file(f, "rb"); on.exit(close(con)); d <- readBin(con, "integer", 2, 4, endian = "little")
  matrix(readBin(con, "double", d[1] * d[2], 8, endian = "little"), d[1], d[2]) }
wr <- function(A, f) { con <- file(f, "wb"); on.exit(close(con)); A <- as.matrix(A)
  writeBin(as.integer(dim(A)), con, 4, endian = "little"); writeBin(as.double(A), con, 8, endian = "little") }
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "tests/fixtures"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
cases <- read.csv("tests/fixtures/cases.csv", stringsAsFactors = FALSE)
for (i in seq_len(nrow(cases))) {
  cs <- cases[i, ]
  M <- rd(file.path("tests/fixtures", sub("\\.csv$", ".bin", cs$file)))
  ks <- as.integer(strsplit(cs$ks, ";")[[1]])
  res <- suppressWarnings(mspca(M, cs$r, ks, type = cs$type, feasibilityConstraintType = cs$ctype,
               verbose = FALSE, maxIter = cs$maxIter, maxRestartTPM = 0, minRestartTPM = 0,
               center = TRUE, scale = as.logical(cs$scale), checkPSD = FALSE))
  wr(res$x_best, file.path(out, paste0(cs$id, "_U.bin")))
  wr(matrix(c(res$objective_value, res$feasibility_violation, res$total_variance, res$runtime), 1),
     file.path(out, paste0(cs$id, "_meta.bin")))
}
cat("done", nrow(cases), "cases on msPCA", as.character(packageVersion("msPCA")), "\n")
