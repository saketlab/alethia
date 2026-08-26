#' Build an embedder backed by a local ONNX model
#'
#' Runs a sentence-embedding model on the CPU with no Python installation, using
#' 'onnxr' for inference, 'tok' for tokenisation, and 'hfhub' to fetch the weights.
#'
#' Pooling reproduces sentence-transformers (an attention-masked mean over token
#' vectors, then L2 normalisation), so embeddings agree with the Python package to
#' floating-point tolerance for the same model.
#'
#' The model must publish ONNX weights. Most sentence-transformers models do, under
#' `onnx/model.onnx`; the `onnx-community` and `Xenova` organisations on Hugging Face
#' publish conversions of many others.
#'
#' @param model_id Hugging Face model id, e.g. `"sentence-transformers/all-MiniLM-L6-v2"`.
#' @param backend Execution provider. `"cpu"` by default; the others are only useful
#'   if you have the matching hardware and an ONNX Runtime built for it.
#' @param threads Inference threads. `NULL` leaves it to ONNX Runtime.
#' @param max_length Sequence length above which inputs are truncated.
#' @param batch_size Texts per forward pass. Lower this if memory is tight; it changes
#'   speed and memory, not results.
#' @param onnx_file Path within the repository to the ONNX weights.
#' @param tokenizer_file Path within the repository to the tokeniser.
#' @return An `alethia_embedder`, usable anywhere a model is accepted.
#' @seealso [use_onnx_backend()] to resolve bare model names through this path.
#' @examples
#' \donttest{
#' # Downloads ~90 MB on first use, then caches.
#' if (onnx_available()) {
#'   embedder <- onnx_embedder("sentence-transformers/all-MiniLM-L6-v2")
#'   alethia(c("Bombay", "Calcutta"), c("Mumbai", "Kolkata"), model = embedder)
#' }
#' }
#' @export
onnx_embedder <- function(model_id,
                          backend = c("cpu", "coreml", "cuda", "xnnpack", "openvino"),
                          threads = NULL,
                          max_length = 256L,
                          batch_size = 32L,
                          onnx_file = "onnx/model.onnx",
                          tokenizer_file = "tokenizer.json") {
  backend <- match.arg(backend)
  require_onnx_stack()

  weights <- hfhub::hub_download(model_id, onnx_file)
  vocab <- hfhub::hub_download(model_id, tokenizer_file)

  args <- list(weights, backend = backend)
  if (!is.null(threads)) {
    args$threads <- as.integer(threads)
  }
  model <- do.call(onnxr::onnx_model, args)

  tokenizer <- tok::tokenizer$from_file(vocab)
  tokenizer$no_padding()
  tokenizer$enable_truncation(as.integer(max_length))

  # not every architecture takes token_type_ids
  wanted <- names(model$input_shapes)

  encode_fn <- function(texts) {
    texts <- as.character(texts)
    chunks <- split(texts, ceiling(seq_along(texts) / batch_size))
    do.call(rbind, lapply(chunks, function(chunk) {
      onnx_forward(model, tokenizer, chunk, wanted)
    }))
  }

  new_embedder(encode_fn, name = model_id, family = paste0("onnx/", backend))
}

# tokenise, pad to the longest row, run, mean-pool, normalise
onnx_forward <- function(model, tokenizer, texts, wanted) {
  encoded <- tokenizer$encode_batch(as.list(texts))
  ids <- lapply(encoded, function(e) e$ids)
  width <- max(vapply(ids, length, integer(1)))

  pad_to <- function(v) c(v, rep(0L, width - length(v)))
  input_ids <- t(vapply(ids, function(v) pad_to(v), numeric(width)))
  attention <- t(vapply(ids, function(v) pad_to(rep(1L, length(v))), numeric(width)))

  as_int_array <- function(m) array(as.integer(m), dim = dim(m))
  inputs <- list(
    input_ids = as_int_array(input_ids),
    attention_mask = as_int_array(attention)
  )
  # built only when declared; otherwise allocated per forward pass
  if ("token_type_ids" %in% wanted) {
    inputs$token_type_ids <- as_int_array(matrix(0, nrow(input_ids), width))
  }
  inputs <- inputs[intersect(wanted, names(inputs))]

  hidden <- do.call(onnxr::onnx_run, c(list(model), inputs))[[1]]

  # padding produces vectors too, making embeddings depend on the longest string
  pooled <- t(vapply(
    seq_len(dim(hidden)[1]),
    function(i) {
      mask <- attention[i, ]
      colSums(hidden[i, , ] * mask) / max(sum(mask), 1e-9)
    },
    numeric(dim(hidden)[3])
  ))
  l2_normalize(pooled)
}

#' Resolve bare model names through the local ONNX runtime
#'
#' After calling this, a model name passed to [alethia()], [cluster_entities()] or
#' [as_embedder()] is loaded and run locally on the CPU.
#'
#' @param ... Passed to [onnx_embedder()], e.g. `threads` or `batch_size`.
#' @return The previously registered backend, invisibly.
#' @examples
#' \donttest{
#' if (onnx_available()) {
#'   use_onnx_backend(threads = 4)
#'   alethia(c("Bombay"), c("Mumbai", "Delhi"), model = "sentence-transformers/all-MiniLM-L6-v2")
#' }
#' }
#' @export
use_onnx_backend <- function(...) {
  require_onnx_stack()
  dots <- list(...)
  # sessions are reused; options are fixed at registration, so the id identifies one
  cache <- new.env(parent = emptyenv())
  set_embedding_backend(function(model_id) {
    if (is.null(cache[[model_id]])) {
      assign(model_id, do.call(onnx_embedder, c(list(model_id), dots)), envir = cache)
    }
    cache[[model_id]]
  })
}

#' Is the local ONNX stack ready to use?
#'
#' Checks that the three R packages are installed and that ONNX Runtime itself has been
#' downloaded. The runtime is a separate one-time download, not an R dependency, so
#' installing the packages is not enough.
#'
#' @return `TRUE` when local ONNX inference will work.
#' @examples
#' onnx_available()
#' @export
onnx_available <- function() {
  all(vapply(
    c("onnxr", "tok", "hfhub"),
    requireNamespace, logical(1),
    quietly = TRUE
  )) && isTRUE(onnxr::onnx_is_installed())
}

require_onnx_stack <- function() {
  missing <- c("onnxr", "tok", "hfhub")
  missing <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop(
      "Local ONNX inference needs: ", paste(missing, collapse = ", "), ".\n",
      "Install with: install.packages(c(",
      paste0('"', missing, '"', collapse = ", "), "))",
      call. = FALSE
    )
  }
  if (!isTRUE(onnxr::onnx_is_installed())) {
    stop(
      "ONNX Runtime is not downloaded yet.\n",
      "Fetch it once with: onnxr::onnx_install()",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
