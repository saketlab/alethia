# Shared helpers for the cross-language parity tests.
#
# char_bag must be bit-for-bit the same function as the one in tools/make_fixtures.py:
# a count of a-z and space. A real transformer would make the fixtures depend on a
# model download and on kernel differences between runtimes.

VOCAB <- c(letters, " ")

char_bag <- function(texts) {
  out <- matrix(0, nrow = length(texts), ncol = length(VOCAB))
  for (row in seq_along(texts)) {
    chars <- strsplit(tolower(as.character(texts[row])), "", fixed = TRUE)[[1]]
    hit <- match(chars, VOCAB)
    hit <- hit[!is.na(hit)]
    if (length(hit)) {
      counts <- tabulate(hit, nbins = length(VOCAB))
      out[row, ] <- counts
    }
  }
  out
}

BIGRAM_DIMS <- 64L

# Second deterministic embedder, matching tools/make_assess_fixture.py. It sees
# character order, which a bag of characters cannot, so it ranks differently.
char_bigram <- function(texts) {
  out <- matrix(0, nrow = length(texts), ncol = BIGRAM_DIMS)
  for (i in seq_along(texts)) {
    idx <- match(strsplit(tolower(as.character(texts[i])), "", fixed = TRUE)[[1]], VOCAB) - 1L
    if (length(idx) < 2) next
    for (j in seq_len(length(idx) - 1L)) {
      a <- idx[j]
      b <- idx[j + 1L]
      if (!is.na(a) && !is.na(b)) {
        slot <- (a * 31L + b) %% BIGRAM_DIMS
        out[i, slot + 1L] <- out[i, slot + 1L] + 1
      }
    }
  }
  out
}

fixture_path <- function(name) {
  testthat::test_path("fixtures", paste0(name, ".json"))
}

load_fixture <- function(name) {
  # jsonlite is only in Suggests, and CRAN runs the tests with suggested packages
  # absent. Skipping keeps that run green without weakening the check when it is there.
  skip_if_not_installed("jsonlite")
  path <- fixture_path(name)
  skip_if_not(file.exists(path), paste("fixture not generated:", name))
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

# Python emits JSON null for NaN and for a missing prediction; R wants NA.
from_json_chr <- function(x) vapply(x, function(v) if (is.null(v)) NA_character_ else as.character(v), character(1))
from_json_num <- function(x) vapply(x, function(v) if (is.null(v)) NA_real_ else as.numeric(v), numeric(1))

# Tolerances, one per data path:
#
# * String scores are rational numbers built from integer edit distances, so the two
#   languages evaluate the same value; only operation ordering can differ, worth at
#   most an ULP or two.
# * The embedding paths run in float64 on both sides, and rank on similarities rounded
#   to RANK_DECIMALS so that equally-similar references are recognised as tied
#   regardless of BLAS build. The remaining slack is ordinary accumulation order.
# * The assessment metrics and the composite run in float64 on both sides.
TOL_STRING <- 1e-12
TOL_FLOAT32 <- 1e-6
TOL_FLOAT64 <- 1e-10
