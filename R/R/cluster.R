#' Group entity strings into clusters of duplicates
#'
#' Mirrors `alethia.cluster.cluster_entities`. Edges require *mutual*
#' nearest-neighbourhood above a cosine floor and each carries a confidence, so no hub
#' chains the whole set into one cluster the way a plain threshold does.
#'
#' @param entities Character vector to group. De-duplicated on exact text first.
#' @param model Anything [as_embedder()] accepts.
#' @param floor Minimum cosine for an edge.
#' @param k Neighbourhood size for the mutual-NN test. Clamped to `[1, n - 1]`.
#' @param require_mutual Require each entity to be in the other's top-k.
#' @param min_confidence Optional cutoff applied to edge confidence before clustering.
#' @param canonical How to name each cluster: `"shortest"` member or `"first"` seen.
#' @return An object of class `alethia_clusters`.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' res <- cluster_entities(c("color", "colour", "zebra"), bag, floor = 0.8)
#' res
#' cluster_records(res)
#' @export
cluster_entities <- function(entities,
                             model,
                             floor = 0.80,
                             k = 5,
                             require_mutual = TRUE,
                             min_confidence = NULL,
                             canonical = c("shortest", "first")) {
  canonical <- match.arg(canonical)
  entities <- unique(as.character(entities))
  n <- length(entities)

  embedder <- as_embedder(model)
  emb <- encode(embedder, entities)

  edges <- mutual_nn_edges(emb, floor = floor, k = k, require_mutual = require_mutual)
  if (!is.null(min_confidence) && nrow(edges)) {
    edges <- edges[edges$confidence >= min_confidence, , drop = FALSE]
  }

  labels <- connected_components(n, edges)
  canon <- canonical_names(entities, labels, canonical)

  structure(
    list(
      entities = entities,
      labels = labels,
      edges = edges,
      canonical = canon,
      embedder_name = embedder$name
    ),
    class = "alethia_clusters"
  )
}

#' Candidate merge edges from mutual nearest neighbours
#'
#' @param emb Numeric matrix, one embedding per row. L2-normalised internally.
#' @param floor Minimum cosine similarity for an edge.
#' @param k Neighbourhood size.
#' @param require_mutual Keep only reciprocated edges.
#' @return A data frame of edges with `i`, `j`, `cosine`, `margin_i`, `margin_j`,
#'   `mutual` and `confidence`, sorted by decreasing confidence.
#' @examples
#' emb <- rbind(c(1, 0), c(0.99, 0.14), c(0, 1))
#' mutual_nn_edges(emb, floor = 0.5, k = 1)
#' @export
mutual_nn_edges <- function(emb, floor = 0.80, k = 5, require_mutual = TRUE) {
  x <- l2_normalize(emb)
  n <- nrow(x)
  if (n < 2L) {
    return(empty_edges())
  }
  sims <- x %*% t(x)
  if (!all(is.finite(sims))) {
    stop("Non-finite similarity matrix; embeddings contain NaN or Inf.", call. = FALSE)
  }

  k <- max(1L, min(as.integer(k), n - 1L))

  # -Inf sorts last and k <= n - 1, so no neighbour is ever the diagonal
  diag(sims) <- -Inf

  neighbours <- vector("list", n)
  margin <- numeric(n)
  for (i in seq_len(n)) {
    row <- sims[i, ]
    # order() is stable, so ties at k resolve by ascending index
    ord <- order(-rank_key(row))
    neighbours[[i]] <- ord[seq_len(k)]
    # raw top two, not ord; a rounding tie could otherwise give a negative margin
    top2 <- sort(row, partial = c(n - 1L, n))[c(n, n - 1L)]
    margin[i] <- top2[1] - top2[2]
  }

  from <- rep(seq_len(n), each = k)
  to <- unlist(neighbours, use.names = FALSE)

  # reciprocity by membership over the n*k directed pairs
  pair_key <- (as.numeric(from) - 1) * n + to
  cosine <- sims[cbind(from, to)]
  mutual <- ((as.numeric(to) - 1) * n + from) %in% pair_key
  keep <- rank_key(cosine) >= rank_key(floor) & (mutual | !require_mutual)

  a <- pmin(from[keep], to[keep])
  b <- pmax(from[keep], to[keep])
  # a pair can be proposed from either end; first past the filters wins
  first <- !duplicated(cbind(a, b))
  if (!any(first)) {
    return(empty_edges())
  }
  a <- a[first]
  b <- b[first]

  margin_i <- margin[a]
  margin_j <- margin[b]
  cosine <- cosine[keep][first]
  edges <- data.frame(
    i = a, j = b, cosine = cosine,
    margin_i = margin_i, margin_j = margin_j,
    mutual = mutual[keep][first],
    confidence = cosine * (1 + pmin(margin_i, margin_j)),
    stringsAsFactors = FALSE
  )
  # total order on (confidence desc, i, j), matching Python's sort key
  edges[order(-edges$confidence, edges$i, edges$j), , drop = FALSE]
}

#' @export
print.alethia_clusters <- function(x, ...) {
  cat(sprintf(
    "<alethia_clusters> %d entities, %d clusters, %d edges (embedder: %s)\n",
    length(x$entities), n_clusters(x), nrow(x$edges), x$embedder_name
  ))
  invisible(x)
}

#' Number of clusters found
#' @param x An `alethia_clusters` object.
#' @return Integer count of distinct clusters.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' n_clusters(cluster_entities(c("color", "colour", "zebra"), bag, floor = 0.8))
#' @export
n_clusters <- function(x) length(unique(x$labels))

#' Cluster membership as a tidy data frame
#' @param x An `alethia_clusters` object.
#' @return A data frame with `entity`, `cluster` and `canonical` columns.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' cluster_records(cluster_entities(c("color", "colour"), bag, floor = 0.8))
#' @export
cluster_records <- function(x) {
  data.frame(
    entity = x$entities,
    cluster = x$labels,
    canonical = x$canonical[x$labels + 1L],
    stringsAsFactors = FALSE
  )
}

# union-find; labels run in first-appearance order, matching Python's
connected_components <- function(n, edges) {
  parent <- seq_len(n)
  find <- function(a) {
    while (parent[a] != a) {
      parent[a] <<- parent[parent[a]]
      a <- parent[a]
    }
    a
  }
  if (nrow(edges)) {
    for (row in seq_len(nrow(edges))) {
      ra <- find(edges$i[row])
      rb <- find(edges$j[row])
      # this frame owns parent; <<- would write past it to the global environment
      if (ra != rb) parent[ra] <- rb
    }
  }
  # unique() and match() are first-appearance ordered
  roots <- vapply(seq_len(n), find, integer(1))
  match(roots, unique(roots)) - 1L
}

# labels run 0..n_clusters-1, so member lab reads at [lab + 1L]
canonical_names <- function(entities, labels, how) {
  vapply(sort(unique(labels)), function(lab) {
    members <- entities[labels == lab]
    if (how == "shortest") {
      # shortest then lexicographic, matching Python's min((len, s))
      shortest <- members[nchar(members) == min(nchar(members))]
      sort(shortest, method = "radix")[1]
    } else {
      members[1]
    }
  }, character(1), USE.NAMES = FALSE)
}

empty_edges <- function() {
  data.frame(
    i = integer(0), j = integer(0), cosine = numeric(0),
    margin_i = numeric(0), margin_j = numeric(0),
    mutual = logical(0), confidence = numeric(0),
    stringsAsFactors = FALSE
  )
}
