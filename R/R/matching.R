#' Align messy entity strings against a reference list
#'
#' The main entry point, mirroring the Python `alethia()`. The pipeline is:
#'
#' 1. drop `NA` and null-like entries, remembering their positions;
#' 2. short-circuit entries that already appear in the reference list;
#' 3. send whatever remains to the chosen backend;
#' 4. splice all three groups back into the caller's original row order.
#'
#' @param dirty_entries Character vector of messy strings.
#' @param reference_entries Character vector of canonical strings.
#' @param model `"rapidfuzz"` for token-sort string similarity (the default, and the
#'   only backend needing no model), or anything [as_embedder()] accepts.
#' @param threshold Minimum score to accept a match; below it the entry is reported as
#'   no match. `NULL` means `DEFAULT_THRESHOLD` for embeddings and no threshold for
#'   string matching. Scores are on `[0, 1]` either way, but string similarity and
#'   cosine measure different things, so a value tuned against one is not calibrated
#'   for the other.
#' @param use_exact_matching Short-circuit entries already present in the references.
#' @param exact_match_case_sensitive Whether that short-circuit respects case.
#' @param drop_duplicates Collapse identical result rows, giving one row per distinct
#'   input. `FALSE` keeps the input length and order, for joining back onto a table.
#' @return A data frame carrying `given_entity`, `alethia_prediction`,
#'   `alethia_score` and `alethia_method`, in input order.
#' @examples
#' alethia(
#'   c("Bombay", "Calcutta", "Mumbai"),
#'   c("Mumbai", "Kolkata", "Chennai")
#' )
#'
#' # Entries already present in the references short-circuit as exact matches, and
#' # null-like values are preserved as no-match rows rather than dropped.
#' alethia(c("Mumbai", NA, "N/A"), c("Mumbai"), drop_duplicates = FALSE)
#' @export
alethia <- function(dirty_entries,
                    reference_entries,
                    model = "rapidfuzz",
                    threshold = NULL,
                    use_exact_matching = TRUE,
                    exact_match_case_sensitive = FALSE,
                    drop_duplicates = TRUE) {
  dirty <- as.character(dirty_entries)
  refs <- filter_null_like(as.character(reference_entries))

  n <- length(dirty)
  prediction <- rep(NA_character_, n)
  score <- rep(NA_real_, n)
  method <- rep(NA_character_, n)

  usable <- !is_null_like(dirty)
  method[!usable] <- "nan"

  if (!length(refs)) {
    # no reference list, so no entry can be matched
    method[usable] <- "no-reference"
    return(finish(dirty, prediction, score, method, drop_duplicates))
  }

  todo <- which(usable)

  if (use_exact_matching && length(todo)) {
    hit <- exact_match_index(dirty[todo], refs, exact_match_case_sensitive)
    matched <- !is.na(hit)
    if (any(matched)) {
      idx <- todo[matched]
      prediction[idx] <- refs[hit[matched]]
      score[idx] <- 1
      method[idx] <- "exact"
      todo <- todo[!matched]
    }
  }

  if (length(todo)) {
    if (identical(model, "rapidfuzz")) {
      res <- match_by_strings(dirty[todo], refs, threshold)
      label <- "rapidfuzz"
    } else {
      # resolve once; for a model name this loads weights and a tokeniser
      embedder <- as_embedder(model)
      res <- match_by_embeddings(
        dirty[todo], refs, embedder,
        threshold %||% DEFAULT_THRESHOLD
      )
      label <- embedder$name
    }
    prediction[todo] <- res$alethia_prediction
    score[todo] <- res$alethia_score
    method[todo] <- label
  }

  finish(dirty, prediction, score, method, drop_duplicates)
}

#' Default minimum cosine similarity for embedding backends
#'
#' String matching does not use this; a cosine and a surface-overlap score of the same
#' value do not mean the same thing.
#' @export
DEFAULT_THRESHOLD <- 0.7

finish <- function(given, prediction, score, method, drop_duplicates) {
  out <- match_results(given, prediction, score)
  out$alethia_method <- method
  if (drop_duplicates) {
    # as pandas' drop_duplicates(): keep the first of each identical row
    out <- out[!duplicated(out), , drop = FALSE]
    rownames(out) <- NULL
  }
  out
}

#' Blank out matches scoring below a threshold
#'
#' Sub-threshold rows are reported as no match. `NULL` leaves the frame untouched.
#'
#' @param results Data frame with `alethia_prediction` and `alethia_score`.
#' @param threshold Minimum acceptable score, or `NULL`.
#' @return The frame, with sub-threshold rows set to `NA`.
#' @examples
#' res <- match_by_strings("Bombay", c("Mumbai", "Kolkata"))
#' apply_threshold(res, 0.9)
#' @export
apply_threshold <- function(results, threshold) {
  if (is.null(threshold) || !nrow(results)) {
    return(results)
  }
  # quantized on both sides, as Python does; a raw compare puts a score within
  # floating-point noise of the cutoff on different sides in different builds
  below <- !is.na(results$alethia_score) &
    rank_key(results$alethia_score) < rank_key(threshold)
  results$alethia_prediction[below] <- NA_character_
  results$alethia_score[below] <- NA_real_
  results
}

# identical to Python's null vocabulary, so both drop the same cells
NULL_LIKE <- c("nan", "null", "none", "", "na", "n/a")

is_null_like <- function(x) {
  is.na(x) | trimws(tolower(as.character(x))) %in% NULL_LIKE
}

filter_null_like <- function(x) x[!is_null_like(x)]

exact_match_index <- function(queries, references, case_sensitive) {
  if (case_sensitive) {
    match(queries, references)
  } else {
    match(tolower(queries), tolower(references))
  }
}

# scalars recycle, so the empty and no-match frames use this same constructor
match_results <- function(given, prediction = NA_character_, score = NA_real_) {
  given <- as.character(given)
  n <- length(given)
  data.frame(
    given_entity = given,
    alethia_prediction = rep_len(as.character(prediction), n),
    alethia_score = rep_len(as.numeric(score), n),
    stringsAsFactors = FALSE
  )
}

empty_results <- function() match_results(character(0))

no_match_results <- function(queries) match_results(queries)

`%||%` <- function(x, y) if (is.null(x)) y else x
