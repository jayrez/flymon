suppressPackageStartupMessages({
  library(malecns)
  library(neuprintr)
})

if (!nzchar(Sys.getenv("NEUPRINT_TOKEN"))) {
  stop("Set NEUPRINT_TOKEN in .env before running the benchmark")
}

Sys.setenv(NEUPRINT_SERVER = "https://neuprint.janelia.org",
           NEUPRINT_DATASET = "male-cns:v1.0")

measure <- function(label, expression) {
  start <- proc.time()
  value <- force(expression)
  elapsed <- (proc.time() - start)[["elapsed"]]
  cat(sprintf("%s: %.3f s\n", label, elapsed))
  value
}

cat("malecns connectome benchmark (network and CPU; no GPU computation)\n")
metadata <- measure("Fetch PN metadata", malecns::mcns_neuprint_meta("/.+_[adl]+PN"))
if (!"bodyid" %in% names(metadata) || nrow(metadata) == 0L) {
  stop("No PN body IDs returned from male-cns:v1.0")
}
bodyids <- head(metadata$bodyid, 3L)
cat(sprintf("Selected %d of %d matching neurons\n", length(bodyids), nrow(metadata)))
neurons <- measure("Fetch and parse 3 skeletons", neuprintr::neuprint_read_neurons(bodyids))
cat(sprintf("Loaded %d skeletons\n", length(neurons)))
