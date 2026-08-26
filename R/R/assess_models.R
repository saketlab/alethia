#' Score candidate embedding models on your own data, without labels
#'
#' Each model is scored on the metrics in [assess-metrics], z-scored **across models**
#' and summed. The composite therefore needs at least two models; with one the score is
#' `NA` and only the component metrics are reported. Weights are normalised **within
#' metric family**, so correlated views of one property cannot count it twice, and a
#' family that could not be computed drops out.
#'
#' @param queries Character vector of dirty/query strings. May be empty, in which case
#'   the retrieval family is skipped and the reference geometry alone is scored.
#' @param references Character vector of canonical strings.
#' @param models Named list of model specifications, each anything [as_embedder()]
#'   accepts. Names are used for reporting.
#' @param weights Optional named numeric vector overriding the per-metric weights.
#'   Interpreted within family when `family_normalize` is `TRUE`.
#' @param family_normalize Rescale weights so each family contributes at most its family
#'   weight. `FALSE` gives flat per-metric weighting.
#' @param variants Optional list of `source` and `variant` character vectors, row `i` a
#'   (clean, dirtied) pair, which enables the robustness family. Generate them with
#'   `alethia.assess.generate_positive_pairs`; omitted, the family drops out.
#' @return An object of class `alethia_assessment` with `$table` (a data frame, best
#'   first) and `$best` (the winning model name, or `NA` when not comparative).
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' rev_bag <- function(texts) {
#'   bag(vapply(texts, function(s) {
#'     paste(rev(strsplit(s, "")[[1]]), collapse = "")
#'   }, character(1)))
#' }
#' refs <- c("aspirin", "ibuprofen", "paracetamol", "metformin", "atenolol")
#' res <- assess_models(
#'   c("asprin", "ibuprofn"), refs,
#'   list(bag = bag, reversed = rev_bag)
#' )
#' res$best
#' @export
assess_models <- function(queries,
                          references,
                          models,
                          weights = NULL,
                          family_normalize = TRUE,
                          variants = NULL) {
  queries <- keep_nonblank(queries)
  references <- keep_nonblank(references)
  if (!length(models)) {
    stop("No models supplied to assess.", call. = FALSE)
  }
  if (is.null(names(models)) || any(!nzchar(names(models)))) {
    stop("`models` must be a named list; names identify each model.", call. = FALSE)
  }
  weights <- if (is.null(weights)) DEFAULT_METRIC_WEIGHTS else weights

  rows <- lapply(names(models), function(name) {
    out <- tryCatch(
      list(
        metrics = assess_one(models[[name]], queries, references, variants),
        error = NA_character_
      ),
      error = function(e) list(metrics = list(), error = conditionMessage(e))
    )
    c(list(model = name), out)
  })

  scores <- composite_scores(rows, weights, family_normalize)

  metric_names <- unique(unlist(lapply(rows, function(r) names(r$metrics))))
  table <- data.frame(
    model = vapply(rows, `[[`, character(1), "model"),
    score = scores, stringsAsFactors = FALSE
  )
  for (metric in metric_names) {
    table[[metric]] <- vapply(
      rows, function(r) {
        v <- r$metrics[[metric]]
        if (is.null(v)) NA_real_ else as.numeric(v)
      }, numeric(1)
    )
  }
  table$error <- vapply(rows, function(r) r$error %||% NA_character_, character(1))
  table <- table[order(-replace(table$score, is.na(table$score), -Inf)), , drop = FALSE]
  rownames(table) <- NULL

  comparative <- sum(is.na(table$error)) >= 2
  best <- if (any(is.finite(table$score))) {
    table$model[which.max(replace(table$score, is.na(table$score), -Inf))]
  } else if (sum(is.na(table$error)) == 1) {
    table$model[is.na(table$error)][1]
  } else {
    NA_character_
  }

  structure(
    list(
      table = table, best = best, is_comparative = comparative,
      n_queries = length(queries), n_references = length(references)
    ),
    class = "alethia_assessment"
  )
}

#' @export
print.alethia_assessment <- function(x, ...) {
  cat(sprintf(
    "<alethia_assessment> %d models on %d queries / %d references\n",
    nrow(x$table), x$n_queries, x$n_references
  ))
  if (!x$is_comparative) {
    cat("  not comparative: the composite needs at least two models\n")
  }
  cat(sprintf("  best: %s\n", x$best))
  print(x$table[, c("model", "score")], row.names = FALSE)
  invisible(x)
}

# a metric absent from ORIENTATION and FAMILY is silently ignored

#' @rdname assess_models
#' @export
DEFAULT_METRIC_WEIGHTS <- c(
  # a shared cone direction moves these while preserving neighbour ordering
  mean_nn_similarity = 0.0,
  confusability_rate = 0.0,
  centered_nn_similarity = 1.0,
  nn_margin_z = 0.0,
  normalized_pr = 0.0,
  alignment_loss = 0.0,
  positive_pair_rank = 1.5,
  uniformity_loss = 0.0,
  mean_margin = 0.0,
  low_margin_rate = 0.0,
  # z-margin tracks distribution shape, so it sits at parity with reciprocity
  mean_margin_z = 1.0,
  low_margin_z_rate = 0.5,
  hubness_skew = 0.75,
  mutual_nn_rate = 1.0
)

METRIC_ORIENTATION <- c(
  mean_nn_similarity = -1, confusability_rate = -1, centered_nn_similarity = -1,
  nn_margin_z = 1, normalized_pr = 1, alignment_loss = -1, positive_pair_rank = 1,
  uniformity_loss = -1, mean_margin = 1, low_margin_rate = -1, mean_margin_z = 1,
  low_margin_z_rate = -1, hubness_skew = -1, mutual_nn_rate = 1
)

METRIC_FAMILY <- c(
  mean_nn_similarity = "separability", confusability_rate = "separability",
  centered_nn_similarity = "separability", nn_margin_z = "separability",
  normalized_pr = "geometry", uniformity_loss = "geometry",
  alignment_loss = "robustness", positive_pair_rank = "robustness",
  mean_margin = "retrieval", low_margin_rate = "retrieval",
  mean_margin_z = "retrieval", low_margin_z_rate = "retrieval",
  mutual_nn_rate = "retrieval", hubness_skew = "pathology"
)

FAMILY_WEIGHTS <- c(
  separability = 2.0,
  robustness = 1.5,
  retrieval = 1.5,
  geometry = 1.0,
  pathology = 0.75
)

keep_nonblank <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  x[nzchar(trimws(x))]
}

assess_one <- function(model, queries, references, variants = NULL) {
  embedder <- as_embedder(model)
  r_emb <- encode(embedder, references)

  # one normalisation and one cosine matrix feed every metric that shares them
  r <- l2_normalize(r_emb)
  out <- c(
    reference_separability_on(r),
    centered_separability(r_emb),
    intrinsic_dimensionality(r_emb),
    list(uniformity_loss = uniformity_loss_on(r))
  )
  if (length(queries)) {
    q <- l2_normalize(encode(embedder, queries))
    sims <- cosine_matrix(q, r)
    out <- c(
      out,
      retrieval_margin_on(sims),
      hubness_on(sims, k = 5),
      # the reverse direction is this matrix transposed
      list(mutual_nn_rate = mutual_nn_rate_on(sims, t(sims), k = 5))
    )
  }
  if (!is.null(variants)) {
    # robustness family; only when the caller supplies pairs
    src_emb <- encode(embedder, variants$source)
    var_emb <- encode(embedder, variants$variant)
    out$alignment_loss <- alignment_loss(src_emb, var_emb)
    out$positive_pair_rank <- positive_pair_rank(src_emb, var_emb)
  }
  out
}

#' Rescale weights so each metric family contributes at most its family weight
#'
#' `available` must list only metrics that vary across the models being compared. A
#' metric identical for every model carries no ranking information; counting it here
#' would let it absorb family weight and then be skipped during scoring, shrinking that
#' family's real influence.
#'
#' @param weights Named numeric vector of per-metric weights.
#' @param available Character vector of metrics that vary across models.
#' @return Named numeric vector of rescaled weights.
#' @keywords internal
family_normalized_weights <- function(weights, available) {
  usable <- names(weights)[names(weights) %in% available & weights > 0]
  if (!length(usable)) {
    return(stats::setNames(numeric(0), character(0)))
  }
  families <- METRIC_FAMILY[usable]
  families[is.na(families)] <- usable[is.na(families)]

  scaled <- numeric(0)
  # first-appearance order, matching Python's accumulation order
  for (family in unique(families)) {
    keys <- usable[families == family]
    total <- sum(weights[keys])
    if (total <= 0) next
    fw <- FAMILY_WEIGHTS[[family]]
    if (is.null(fw) || is.na(fw)) fw <- 1.0
    scaled[keys] <- weights[keys] / total * fw
  }
  scaled
}

composite_scores <- function(rows, weights, family_normalize) {
  n <- length(rows)
  scores <- rep(NA_real_, n)
  valid <- which(vapply(rows, function(r) is.na(r$error %||% NA_character_), logical(1)))
  if (length(valid) < 2) {
    return(scores)
  }
  scores[valid] <- 0

  metric_value <- function(idx, key) {
    v <- rows[[idx]]$metrics[[key]]
    if (is.null(v)) NA_real_ else as.numeric(v)
  }

  if (family_normalize) {
    all_keys <- unique(unlist(lapply(rows[valid], function(r) names(r$metrics))))
    available <- character(0)
    for (key in all_keys) {
      vals <- vapply(valid, metric_value, numeric(1), key = key)
      finite <- vals[is.finite(vals)]
      # only a metric that differs between models can rank them
      if (length(finite) >= 2 && stats::sd(finite) > 0) {
        available <- c(available, key)
      }
    }
    weights <- family_normalized_weights(weights, sort(available))
  }

  for (key in names(weights)) {
    vals <- vapply(valid, metric_value, numeric(1), key = key)
    # pathological in either direction; only magnitude ranks
    if (identical(key, "hubness_skew")) vals <- abs(vals)
    finite <- is.finite(vals)
    if (sum(finite) < 2) next

    orientation <- METRIC_ORIENTATION[[key]]
    # impute at the worst observed value, so failing cannot beat scoring badly
    worst <- if (orientation > 0) min(vals[finite]) else max(vals[finite])
    vals[!finite] <- worst

    mu <- mean(vals)
    sigma <- sqrt(mean((vals - mu)^2))
    if (sigma == 0) next
    z <- (vals - mu) / sigma * orientation * weights[[key]]
    scores[valid] <- scores[valid] + z
  }
  scores
}
