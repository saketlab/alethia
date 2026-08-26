#' Decimal places similarities are rounded to before any ranking decision
#'
#' Equally similar references are common in real data: shared tokens, anagrams,
#' duplicated records. Whether such a pair is *seen* as tied otherwise depends on the
#' last bits of a dot product, which differ between BLAS builds and languages. Rounding
#' first makes "equally similar" mean the same thing everywhere, so the tie-break by
#' ascending index gives the same answer. Reported scores are left unrounded.
#' @export
RANK_DECIMALS <- 12

#' Similarities quantized for a reproducible ordering
#' @param sims Numeric vector or matrix of similarities.
#' @return The input rounded to [RANK_DECIMALS] places.
#' @keywords internal
rank_key <- function(sims) round(sims, RANK_DECIMALS)

#' Row-wise L2 normalisation
#'
#' The single canonical normalisation used by matching, clustering and assessment,
#' mirroring `alethia.embedder.l2_normalize`. Zero rows stay zero, never
#' producing `NaN`.
#'
#' @param x Numeric matrix, one embedding per row.
#' @return Matrix of the same shape with unit-norm rows.
#' @examples
#' l2_normalize(rbind(c(3, 4), c(0, 0)))
#' @export
l2_normalize <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  norms <- sqrt(rowSums(x * x))
  norms[norms == 0] <- 1
  x / norms
}

#' Coerce a model specification into an embedder
#'
#' Mirrors `alethia.embedder.as_embedder`. An embedder is any object with an
#' `encode(texts)` method returning an `n x d` numeric matrix. This function accepts:
#'
#' * a function `f(texts) -> matrix`, wrapped as a callable embedder;
#' * an object already carrying an `encode` element (returned unchanged);
#' * a character model name, resolved through the registered backend (see
#'   [set_embedding_backend()]).
#'
#' Every embedding source sits behind this seam, so matching, clustering and assessment
#' share one implementation.
#'
#' @param model A function, an embedder, or a model name.
#' @param name Optional display name.
#' @return An object of class `alethia_embedder`.
#' @examples
#' # Any function from character vector to matrix is an embedder.
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' e <- as_embedder(bag, name = "letter-count")
#' e
#' @export
as_embedder <- function(model, name = NULL) {
  if (inherits(model, "alethia_embedder")) {
    return(model)
  }
  if (is.function(model)) {
    return(new_embedder(model, name %||% "callable", "callable"))
  }
  if (is.character(model) && length(model) == 1L) {
    if (model %in% NON_EMBEDDING_MODELS) {
      stop(
        sprintf(
          "'%s' is not an embedding model; it is handled by the string-similarity path.",
          model
        ),
        call. = FALSE
      )
    }
    backend <- get_embedding_backend()
    if (is.null(backend)) {
      stop(
        "No embedding backend is registered, so a model name cannot be resolved.\n",
        "Register one with set_embedding_backend(), or pass a function that takes a\n",
        "character vector and returns a matrix.",
        call. = FALSE
      )
    }
    resolved <- backend(model)
    # a ready-made embedder keeps the name and family the backend set
    if (inherits(resolved, "alethia_embedder")) {
      return(resolved)
    }
    return(new_embedder(resolved, name %||% model, "backend"))
  }
  stop(
    sprintf(
      "Cannot interpret an object of class '%s' as an embedder.",
      paste(class(model), collapse = "/")
    ),
    call. = FALSE
  )
}

NON_EMBEDDING_MODELS <- c("rapidfuzz", "exact", "fuzzy", "instructor")

new_embedder <- function(encode, name, family) {
  structure(
    list(encode = encode, name = name, family = family),
    class = "alethia_embedder"
  )
}

#' @export
print.alethia_embedder <- function(x, ...) {
  cat(sprintf("<alethia_embedder> name=%s family=%s\n", x$name, x$family))
  invisible(x)
}

#' Encode text with an embedder
#'
#' @param embedder An `alethia_embedder`, from [as_embedder()].
#' @param texts Character vector.
#' @return Numeric matrix with one row per input.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' dim(encode(as_embedder(bag), c("colour", "color")))
#' @export
encode <- function(embedder, texts) {
  embedder <- as_embedder(embedder)
  texts <- as.character(texts)
  if (!length(texts)) {
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }
  out <- embedder$encode(texts)
  out <- as.matrix(out)
  storage.mode(out) <- "double"
  if (nrow(out) != length(texts)) {
    stop(
      sprintf(
        "Embedder '%s' returned %d rows for %d inputs.",
        embedder$name, nrow(out), length(texts)
      ),
      call. = FALSE
    )
  }
  if (!all(is.finite(out))) {
    stop(
      sprintf("Embedder '%s' produced non-finite embeddings (NaN or Inf).", embedder$name),
      call. = FALSE
    )
  }
  out
}

.alethia_state <- new.env(parent = emptyenv())

#' Register a backend that turns model names into embedding functions
#'
#' The package ships no model runtime of its own. Point this at whatever you use to
#' produce embeddings, e.g. a local ONNX session, an API client, or a cached matrix.
#'
#' @param backend A function taking a model name and returning a function
#'   `f(texts) -> matrix`. `NULL` clears the registration.
#' @return The previous backend, invisibly.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' old <- set_embedding_backend(function(model_name) bag)
#' as_embedder("any-model-name")
#' set_embedding_backend(old)
#' @export
set_embedding_backend <- function(backend) {
  previous <- .alethia_state$backend
  .alethia_state$backend <- backend
  invisible(previous)
}

#' @rdname set_embedding_backend
#' @export
get_embedding_backend <- function() .alethia_state$backend

#' Match queries to references by embedding cosine similarity
#'
#' The single embedding-based matching primitive, mirroring
#' `alethia.embedder.match_by_embeddings`. Every embedding backend routes through it.
#'
#' @param queries Character vector of strings to match.
#' @param references Character vector of candidate strings.
#' @param embedder Anything [as_embedder()] accepts.
#' @param threshold Optional minimum cosine. Matches below it become `NA`.
#' @return A data frame with `given_entity`, `alethia_prediction`, `alethia_score`.
#' @examples
#' bag <- function(x) {
#'   t(sapply(
#'     strsplit(tolower(x), ""),
#'     function(ch) tabulate(match(ch, letters), 26)
#'   ))
#' }
#' match_by_embeddings(c("colour"), c("color", "zebra"), bag)
#' @export
match_by_embeddings <- function(queries, references, embedder, threshold = NULL) {
  queries <- as.character(queries)
  references <- as.character(references)
  if (!length(queries)) {
    return(empty_results())
  }
  if (!length(references)) {
    return(no_match_results(queries))
  }

  embedder <- as_embedder(embedder)
  q <- l2_normalize(encode(embedder, queries))
  r <- l2_normalize(encode(embedder, references))
  sims <- q %*% t(r)

  # choose on rounded values, report the unrounded score
  best <- max.col(rank_key(sims), ties.method = "first")
  scores <- sims[cbind(seq_len(nrow(sims)), best)]

  apply_threshold(
    match_results(queries, references[best], as.numeric(scores)),
    threshold
  )
}
