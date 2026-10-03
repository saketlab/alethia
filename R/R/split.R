#' Match composite entries by splitting them on a separator first
#'
#' Mirrors the Python `alethia_split()`. Each entry is split into pieces on a literal
#' separator, every piece is matched independently against `reference_entries` via
#' [alethia()], and the highest-scoring piece becomes the entry's prediction. Ties
#' break on first occurrence, same as the rest of the package. Every piece's own
#' match rides along in `alethia_parts`.
#'
#' @param dirty_entries Character vector of messy strings.
#' @param reference_entries Character vector of canonical strings.
#' @param split_on Literal separator each entry is split on (e.g. `","`, `";"`,
#'   `" and "`). Entries with no separator, or that split to nothing usable, are
#'   matched whole.
#' @param ... Forwarded to [alethia()] for the per-piece matching (`model`,
#'   `threshold`, ...).
#' @return A data frame of `given_entity`, `alethia_prediction`, `alethia_score`,
#'   `alethia_matched_part` (the winning piece), and `alethia_parts` (a list column:
#'   every piece tried, as a list of `part`/`prediction`/`score`, best first).
#' @examples
#' alethia_split(
#'   c("CNS Tuberculosis, Septic Shock"),
#'   c("Tuberculosis of nervous system", "Septic shock"),
#'   split_on = ","
#' )
#' @export
alethia_split <- function(dirty_entries, reference_entries, split_on, ...) {
  dirty_entries <- as.character(dirty_entries)
  parts_list <- lapply(dirty_entries, split_parts, split_on = split_on)

  flat_parts <- unlist(parts_list, use.names = FALSE)

  # a repeated piece is matched once
  uniques <- unique(flat_parts)
  codes <- match(flat_parts, uniques)
  part_results <- alethia(uniques, reference_entries, ...)
  codes_by_entry <- split(codes, rep(seq_along(parts_list), lengths(parts_list)))

  n <- length(dirty_entries)
  prediction <- rep(NA_character_, n)
  score <- rep(NA_real_, n)
  matched_part <- rep(NA_character_, n)
  parts_col <- vector("list", n)

  for (i in seq_len(n)) {
    rows <- codes_by_entry[[i]]
    ord <- order(-rank_key(part_results$alethia_score[rows]))
    best <- rows[ord[1]]

    prediction[i] <- part_results$alethia_prediction[best]
    score[i] <- part_results$alethia_score[best]
    matched_part[i] <- part_results$given_entity[best]

    parts_col[[i]] <- lapply(rows[ord], function(r) {
      list(
        part = part_results$given_entity[r],
        prediction = part_results$alethia_prediction[r],
        score = part_results$alethia_score[r]
      )
    })
  }

  out <- data.frame(
    given_entity = dirty_entries,
    alethia_prediction = prediction,
    alethia_score = score,
    alethia_matched_part = matched_part,
    stringsAsFactors = FALSE
  )
  out$alethia_parts <- parts_col
  out
}

split_parts <- function(entry, split_on) {
  if (is_null_like(entry)) {
    return(entry)
  }
  parts <- trimws(strsplit(entry, split_on, fixed = TRUE)[[1]])
  parts <- parts[nzchar(parts)]
  if (!length(parts)) parts <- entry
  parts
}
