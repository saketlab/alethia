#' Top-1 accuracy and MRR of an embedder against ground truth
#'
#' The labelled counterpart to the label-free metrics: given the correct reference for
#' each query, how often does the model retrieve it? [assess_models()] predicts this
#' without labels.
#'
#' @param queries Character vector of query strings.
#' @param references Character vector of candidate strings.
#' @param truth Correct reference per query, or `NA` for an open-world NIL query with no
#'   correct answer. Real matching input is full of these: a cause of death that is not
#'   a diagnosis, a drug outside the formulary.
#' @param model Anything [as_embedder()] accepts.
#' @param nil_threshold Cosine below which a top-1 match counts as an abstention.
#'   Required when `truth` contains `NA`; ignored otherwise.
#' @return A named list with `top1` and `mrr`, plus `nil_recall` and `answerable_top1`
#'   when the dataset contains NIL queries. Those two are separate because one accuracy
#'   number hides the trade-off: a model can look good by always abstaining, or never.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' true_accuracy(
#'   c("colour", "flavour"), c("color", "flavor"),
#'   c("color", "flavor"), bag
#' )
#' @export
true_accuracy <- function(queries, references, truth, model, nil_threshold = NULL) {
  queries <- as.character(queries)
  references <- as.character(references)
  truth <- as.character(truth)

  if (!length(queries) || !length(references)) {
    return(list(top1 = NA_real_, mrr = NA_real_))
  }

  has_nil <- anyNA(truth)
  if (has_nil && is.null(nil_threshold)) {
    stop(
      "truth contains NIL (NA) entries, so nil_threshold is required: without it there ",
      "is no rule by which the model can decline to answer.",
      call. = FALSE
    )
  }

  embedder <- as_embedder(model)
  q <- l2_normalize(encode(embedder, queries))
  r <- l2_normalize(encode(embedder, references))
  sims <- q %*% t(r)

  ref_index <- stats::setNames(seq_along(references), references)
  n <- length(queries)
  top1 <- 0L
  rr_sum <- 0
  n_answerable <- 0L
  answerable_hits <- 0L
  nil_total <- 0L
  nil_hits <- 0L

  for (i in seq_len(n)) {
    row <- sims[i, ]
    best_sim <- max(row)
    abstained <- has_nil && best_sim < nil_threshold
    correct <- truth[i]

    if (is.na(correct)) {
      nil_total <- nil_total + 1L
      if (abstained) {
        top1 <- top1 + 1L
        nil_hits <- nil_hits + 1L
      }
      next
    }

    n_answerable <- n_answerable + 1L
    gold <- ref_index[[correct]]
    # rank without sorting: those strictly better, plus ties at a lower index
    key <- rank_key(row)
    rank <- sum(key > key[gold]) + sum(key[seq_len(gold - 1L)] == key[gold]) + 1L
    if (rank == 1L && !abstained) {
      top1 <- top1 + 1L
      answerable_hits <- answerable_hits + 1L
    }
    rr_sum <- rr_sum + 1 / rank
  }

  out <- list(
    top1 = top1 / n,
    mrr = if (n_answerable) rr_sum / n_answerable else NA_real_
  )
  if (has_nil) {
    out$nil_recall <- if (nil_total) nil_hits / nil_total else NA_real_
    out$answerable_top1 <- if (n_answerable) answerable_hits / n_answerable else NA_real_
  }
  out
}
