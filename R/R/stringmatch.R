#' Token-sort similarity ratio
#'
#' Reproduces `rapidfuzz.fuzz.token_sort_ratio`, the scorer behind the default
#' `"rapidfuzz"` backend.
#'
#' Exact agreement rests on an identity: RapidFuzz's `ratio` is the normalised indel
#' (insert/delete-only edit) similarity,
#'
#'   ratio(a, b) = 100 * (1 - indel(a, b) / (nchar(a) + nchar(b)))
#'
#' and `stringdist(method = "lcs")` computes that indel distance. Both sides evaluate
#' the same rational number in double precision, so the scores agree to within an ULP.
#'
#' Tokens are split on whitespace, sorted, and rejoined with a single space. Case and
#' punctuation are left alone, so `"bombay"` and `"Bombay"` score 83.33, not 100.
#'
#' @param a,b Character vectors, recycled against each other.
#' @return Numeric vector of scores in `[0, 100]`.
#' @examples
#' # Token order does not matter, but case does.
#' token_sort_ratio("New Delhi", "Delhi New")
#' token_sort_ratio("Bombay", "Mumbai")
#' @export
token_sort_ratio <- function(a, b) {
  100 * indel_ratio(token_sort(a), token_sort(b))
}

# total is a parameter so match_by_strings() can hoist lengths out of its loop
indel_ratio <- function(a, b, total = nchar(a) + nchar(b)) {
  d <- stringdist::stringdist(a, b, method = "lcs")
  # two empty strings are identical; the ratio is 0/0 there
  ifelse(total == 0, 1, 1 - d / total)
}

#' Normalise a string by sorting its whitespace-separated tokens
#'
#' The preprocessing step inside [token_sort_ratio()], exported because the same
#' normalisation is useful when caching or de-duplicating inputs.
#'
#' @param x Character vector.
#' @return Character vector with tokens sorted and single-space separated.
#' @examples
#' token_sort("  South 24 Parganas ")
#' @export
token_sort <- function(x) {
  x <- as.character(x)
  vapply(
    strsplit(x, "[ \t\r\n]+"),
    function(tokens) {
      tokens <- tokens[nzchar(tokens)]
      # a locale-dependent order makes non-ASCII scores machine-dependent
      paste(sort(tokens, method = "radix"), collapse = " ")
    },
    character(1)
  )
}

#' Match strings against a reference list by token-sort similarity
#'
#' The counterpart of Python's `run_rapidfuzz_matching()`: for each query, the
#' highest-scoring reference.
#'
#' @param queries Character vector of strings to match.
#' @param references Character vector of candidate strings.
#' @param threshold Minimum score in `[0, 1]`; below it an entry is `NA`. `NULL`
#'   returns the best match whatever it scores.
#' @return A data frame with `given_entity`, `alethia_prediction` and `alethia_score`.
#'   Scores are rescaled to `[0, 1]`.
#' @examples
#' match_by_strings(
#'   c("Bombay", "Calcutta"),
#'   c("Mumbai", "Kolkata", "Chennai")
#' )
#'
#' # Demand a close match, and report anything weaker as no match.
#' match_by_strings("Bombay", c("Mumbai", "Kolkata"), threshold = 0.9)
#' @export
match_by_strings <- function(queries, references, threshold = NULL) {
  queries <- as.character(queries)
  references <- as.character(references)

  if (!length(queries)) {
    return(empty_results())
  }
  if (!length(references)) {
    return(no_match_results(queries))
  }

  ref_sorted <- token_sort(references)
  ref_len <- nchar(ref_sorted)
  q_sorted <- token_sort(queries)
  q_len <- nchar(q_sorted)

  preds <- character(length(queries))
  scores <- numeric(length(queries))
  for (i in seq_along(queries)) {
    s <- indel_ratio(q_sorted[i], ref_sorted, total = q_len[i] + ref_len)
    best <- which.max(s)
    preds[i] <- references[best]
    scores[i] <- s[best]
  }

  apply_threshold(match_results(queries, preds, scores), threshold)
}
