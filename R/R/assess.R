#' Label-free geometric metrics for an embedding set
#'
#' These score an embedding model on your own data without ground-truth labels, by
#' measuring the geometry it produces. Ports of `alethia.assess.metrics`; each formula
#' is stated so the two implementations can be checked against one another.
#'
#' @name assess-metrics
NULL

#' Reference-set separability
#'
#' How well distinct references are kept apart. Returns the mean and median cosine to
#' the nearest *other* reference (lower is better), and the fraction whose nearest
#' other reference exceeds `confusion_threshold`.
#'
#' These are raw cosines, so they partly reflect anisotropy: transformer embeddings
#' occupy a narrow cone, pushing every cosine high. Retrieval depends on the ordering
#' of similarities, not their level, so [centered_separability()] gives the
#' anisotropy-corrected view.
#'
#' @param ref_emb Numeric matrix of reference embeddings.
#' @param confusion_threshold Cosine above which two references count as confusable.
#' @return Named list of `mean_nn_similarity`, `median_nn_similarity`,
#'   `confusability_rate`.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' reference_separability(refs)
#' @export
reference_separability <- function(ref_emb, confusion_threshold = 0.9) {
  reference_separability_on(l2_normalize(ref_emb), confusion_threshold)
}

reference_separability_on <- function(x, confusion_threshold = 0.9) {
  n <- nrow(x)
  if (n < 2L) {
    return(list(
      mean_nn_similarity = 0, median_nn_similarity = 0, confusability_rate = 0
    ))
  }
  nn <- nearest_other_similarity(x)
  list(
    mean_nn_similarity = mean(nn),
    median_nn_similarity = stats::median(nn),
    confusability_rate = mean(nn > confusion_threshold)
  )
}

#' Anisotropy-corrected separability
#'
#' [reference_separability()] after removing the set mean, which strips the shared
#' direction responsible for uniformly high cosines.
#'
#' @param ref_emb Numeric matrix of reference embeddings.
#' @return Named list of `centered_nn_similarity` and `nn_margin_z`.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' centered_separability(refs)
#' @export
centered_separability <- function(ref_emb) {
  x <- as.matrix(ref_emb)
  storage.mode(x) <- "double"
  n <- nrow(x)
  if (n < 3L) {
    return(list(centered_nn_similarity = 0, nn_margin_z = 0))
  }
  centered <- l2_normalize(sweep(x, 2, colMeans(x), "-"))
  sims <- centered %*% t(centered)
  # NA masks the diagonal for both uses without a second n x n copy
  diag(sims) <- NA_real_
  nn <- apply(sims, 1, max, na.rm = TRUE)

  mu <- rowMeans(sims, na.rm = TRUE)
  # population SD over the n-1 others, matching numpy's ddof = 0
  sd_row <- sqrt(rowMeans((sims - mu)^2, na.rm = TRUE))
  z <- (nn - mu) / sd_row
  # a degenerate row has no scale to express a margin in
  z[sd_row == 0] <- NA_real_
  list(centered_nn_similarity = mean(nn), nn_margin_z = mean(z, na.rm = TRUE))
}

#' Intrinsic dimensionality
#'
#' Participation ratio `(sum(lambda))^2 / sum(lambda^2)` over the PCA eigenvalues, and
#' the same divided by the ambient dimension. Higher means the model spreads the data
#' over more directions.
#'
#' @param emb Numeric matrix of embeddings.
#' @return Named list of `participation_ratio` and `normalized_pr`.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' intrinsic_dimensionality(refs)
#' @export
intrinsic_dimensionality <- function(emb) {
  x <- as.matrix(emb)
  storage.mode(x) <- "double"
  if (nrow(x) < 2L) {
    return(list(participation_ratio = 0, normalized_pr = 0))
  }
  centered <- sweep(x, 2, colMeans(x), "-")
  lam <- eigen(crossprod(centered), symmetric = TRUE, only.values = TRUE)$values
  lam <- pmax(lam, 0)
  total <- sum(lam)
  if (total <= 0) {
    return(list(participation_ratio = 0, normalized_pr = 0))
  }
  pr <- total^2 / sum(lam^2)
  list(participation_ratio = pr, normalized_pr = pr / ncol(x))
}

#' Alignment loss
#'
#' Mean `||x - y||^alpha` over matched positive pairs, on L2-normalised embeddings.
#' Lower means the model keeps a string and its dirtied variant close. After
#' Wang & Isola (2020).
#'
#' @param src_emb,var_emb Matched embedding matrices, row `i` a positive pair.
#' @param alpha Exponent on the distance.
#' @return Scalar loss.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' alignment_loss(refs, queries)
#' @export
alignment_loss <- function(src_emb, var_emb, alpha = 2) {
  a <- l2_normalize(src_emb)
  b <- l2_normalize(var_emb)
  if (!nrow(a) || nrow(a) != nrow(b)) {
    return(NA_real_)
  }
  d <- sqrt(pmax(0, rowSums((a - b)^2)))
  mean(d^alpha)
}

#' Positive-pair rank
#'
#' How highly each synthesized positive ranks among *all* variants. `src_emb[i]` and
#' `var_emb[i]` are a positive pair; for each source this is the percentile rank of its
#' own variant among every variant, averaged. Higher is better. Ties take their
#' mid-rank.
#'
#' Unlike [alignment_loss()], this uses only the *ordering* of each source's
#' similarities, so it is invariant to any strictly increasing transformation of cosine,
#' including the affine shift from adding a common cone direction and renormalising.
#' Hence its weight in the composite.
#'
#' @param src_emb,var_emb Matched embedding matrices, row `i` a positive pair.
#' @return Scalar mean percentile rank in `[0, 1]`.
#' @family assess metrics
#' @export
positive_pair_rank <- function(src_emb, var_emb) {
  src <- as.matrix(src_emb)
  var <- as.matrix(var_emb)
  if (!nrow(src) || !nrow(var)) {
    return(NA_real_)
  }
  if (nrow(src) != nrow(var)) {
    stop("src_emb and var_emb must have the same number of rows", call. = FALSE)
  }
  src <- l2_normalize(src)
  var <- l2_normalize(var)
  n <- nrow(src)
  # an exact == misses equal similarities, shifting mid-ranks between BLAS builds
  sims <- rank_key(src %*% t(var))
  positive <- sims[cbind(seq_len(n), seq_len(n))]
  # mid-rank for ties; positive recycles down columns
  lower <- rowSums(sims < positive)
  equal <- rowSums(sims == positive)
  mean((lower + 0.5 * equal) / n)
}

#' Uniformity loss
#'
#' `log E exp(-t * ||x - y||^2)` over all pairs. More negative means the embeddings
#' spread more evenly over the sphere, wasting less of the representation. After
#' Wang & Isola (2020).
#'
#' @param emb Numeric matrix of embeddings.
#' @param t Temperature.
#' @param max_points Subsample size when there are more rows than this.
#' @return Scalar loss.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' uniformity_loss(refs)
#' @export
uniformity_loss <- function(emb, t = 2, max_points = 2000) {
  uniformity_loss_on(l2_normalize(emb), t, max_points)
}

# normalisation is row-wise, so subsampling after gives the same rows
uniformity_loss_on <- function(x, t = 2, max_points = 2000) {
  n <- nrow(x)
  if (n < 2L) {
    return(NA_real_)
  }
  if (n > max_points) {
    # even stride, matching numpy's linspace subsample
    idx <- as.integer(seq(0, n - 1, length.out = max_points)) + 1L
    x <- x[idx, , drop = FALSE]
    n <- nrow(x)
  }
  gram <- x %*% t(x)
  # ||x - y||^2 = 2 - 2 x.y on unit rows, avoiding the difference tensor
  total <- 0
  for (row in seq_len(n - 1L)) {
    cols <- (row + 1L):n
    pdist <- pmax(0, 2 - 2 * gram[row, cols])
    total <- total + sum(exp(-t * pdist))
  }
  log(total / (n * (n - 1) / 2))
}

#' Retrieval margin
#'
#' How decisively the top reference beats the runner-up for each query, both in raw
#' cosine and in per-query standard deviations (`_z`, which is scale-free and so
#' comparable across models with different cosine ranges).
#'
#' @param query_emb,ref_emb Numeric embedding matrices.
#' @param low_margin Margin below which a match counts as ambiguous.
#' @return Named list of `mean_margin`, `low_margin_rate`, `mean_margin_z`,
#'   `low_margin_z_rate`.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' retrieval_margin(queries, refs)
#' @export
retrieval_margin <- function(query_emb, ref_emb, low_margin = 0.02) {
  retrieval_margin_on(
    cosine_matrix(l2_normalize(query_emb), l2_normalize(ref_emb)),
    low_margin
  )
}

retrieval_margin_on <- function(sims, low_margin = 0.02) {
  if (ncol(sims) < 2L) {
    # no runner-up; Python reports these constants, not NA
    return(list(
      mean_margin = 0, low_margin_rate = 1,
      mean_margin_z = 0, low_margin_z_rate = 1
    ))
  }
  if (!nrow(sims)) {
    return(list(
      mean_margin = NaN, low_margin_rate = NaN,
      mean_margin_z = NaN, low_margin_z_rate = NaN
    ))
  }
  # two masked maxima, not a row sort; only values are used
  rows <- seq_len(nrow(sims))
  top1 <- max.col(sims, ties.method = "first")
  rest <- sims
  rest[cbind(rows, top1)] <- -Inf
  runner_up <- max.col(rest, ties.method = "first")
  margin <- sims[cbind(rows, top1)] - rest[cbind(rows, runner_up)]

  mu <- rowMeans(sims)
  sd_row <- sqrt(rowMeans((sims - mu)^2))
  margin_z <- margin / sd_row
  # zero SD leaves the z-margin undefined, as in centered_separability()
  margin_z[sd_row == 0] <- NA_real_
  list(
    mean_margin = mean(margin),
    low_margin_rate = mean(margin < low_margin),
    mean_margin_z = mean(margin_z, na.rm = TRUE),
    low_margin_z_rate = mean(!is.na(margin_z) & margin_z < 1)
  )
}

#' Hubness
#'
#' Skewness of the k-occurrence distribution: how unevenly references are drawn as
#' top-k neighbours. High values are pathological, a few "hub" references absorbing
#' many queries. After Radovanovic et al. (2010), JMLR 11:2487-2531.
#'
#' @param query_emb,ref_emb Numeric embedding matrices.
#' @param k Neighbourhood size.
#' @return Named list with `hubness_skew`.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' hubness(queries, refs, k = 2)
#' @export
hubness <- function(query_emb, ref_emb, k = 5) {
  hubness_on(cosine_matrix(l2_normalize(query_emb), l2_normalize(ref_emb)), k)
}

hubness_on <- function(sims, k = 5) {
  n_ref <- ncol(sims)
  k <- min(k, n_ref)
  if (k < 1L || !nrow(sims)) {
    return(list(hubness_skew = 0))
  }
  counts <- numeric(n_ref)
  for (row in seq_len(nrow(sims))) {
    topk <- order(-rank_key(sims[row, ]))[seq_len(k)] # ties by ascending index
    counts[topk] <- counts[topk] + 1
  }
  mu <- mean(counts)
  sigma <- sqrt(mean((counts - mu)^2))
  if (sigma == 0) {
    return(list(hubness_skew = 0))
  }
  list(hubness_skew = mean(((counts - mu) / sigma)^3))
}

#' Mutual nearest-neighbour rate
#'
#' Fraction of query-to-reference top-1 assignments reciprocated within the
#' reference's own top-k. Higher means more stable, more trustworthy matches.
#'
#' @param query_emb,ref_emb Numeric embedding matrices.
#' @param k Neighbourhood size on the reference side.
#' @return Scalar rate.
#' @family assess metrics
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' refs <- bag(c("aspirin", "ibuprofen", "paracetamol", "metformin"))
#' queries <- bag(c("asprin", "ibuprofn", "paracetmol", "metfornin"))
#' mutual_nn_rate(queries, refs, k = 2)
#' @export
mutual_nn_rate <- function(query_emb, ref_emb, k = 5) {
  q <- l2_normalize(query_emb)
  r <- l2_normalize(ref_emb)
  mutual_nn_rate_on(cosine_matrix(q, r), cosine_matrix(r, q), k)
}

# back is the reference-to-query matrix, supplied by the call site
mutual_nn_rate_on <- function(sims, back, k = 5) {
  nq <- nrow(sims)
  if (!nq || !ncol(sims)) {
    return(NA_real_)
  }
  nr <- ncol(sims)
  top1 <- max.col(rank_key(sims), ties.method = "first")
  kk <- min(k, nq)
  reciprocated <- logical(nq)
  # a reference's top-k does not depend on which query asked
  topk_of <- vector("list", nr)
  for (i in seq_len(nq)) {
    ref <- top1[i]
    if (is.null(topk_of[[ref]])) {
      topk_of[[ref]] <- order(-rank_key(back[ref, ]))[seq_len(kk)] # ties by index
    }
    reciprocated[i] <- i %in% topk_of[[ref]]
  }
  mean(reciprocated)
}

# nearest-other-reference cosine, diagonal masked
nearest_other_similarity <- function(x) {
  sims <- x %*% t(x)
  diag(sims) <- -Inf
  apply(sims, 1, max)
}

# encode() returns 0 x 0 for empty input, which %*% rejects as non-conformable
cosine_matrix <- function(q, r) {
  if (!nrow(q) || !nrow(r) || !ncol(q)) {
    return(matrix(numeric(0), nrow(q), nrow(r)))
  }
  q %*% t(r)
}
