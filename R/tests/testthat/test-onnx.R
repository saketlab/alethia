# Local ONNX inference. Everything here needs the runtime and a model download, so it
# skips on CRAN and on any machine without the stack. The pooling assertion is the one
# that matters: it pins R's output to what sentence-transformers produces in Python, so
# embeddings from the two packages stay interchangeable.

skip_onnx <- function() {
  skip_on_cran()
  skip_if_not(onnx_available(), "ONNX runtime or model stack not available")
}

MODEL <- "sentence-transformers/all-MiniLM-L6-v2"

test_that("onnx_available reports without erroring", {
  expect_type(onnx_available(), "logical")
})

test_that("a missing stack fails with the exact install command", {
  # Reaching the guard does not require the stack to be absent; the message must name
  # what to run either way.
  if (onnx_available()) {
    expect_true(require_onnx_stack())
  } else {
    expect_error(require_onnx_stack(), "install.packages|onnx_install")
  }
})

test_that("the embedder produces unit-norm rows of the model's width", {
  skip_onnx()
  e <- onnx_embedder(MODEL)
  emb <- encode(e, c("Bombay", "Mumbai", "aspirin"))
  expect_equal(nrow(emb), 3)
  expect_equal(ncol(emb), 384)
  expect_equal(sqrt(rowSums(emb^2)), rep(1, 3), tolerance = 1e-6)
})

test_that("pooling matches sentence-transformers to float32 tolerance", {
  skip_onnx()
  # Reference values from Python:
  #   SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
  #     .encode(["Bombay", "Mumbai", "aspirin"], normalize_embeddings=True)[:, :6]
  expected <- rbind(
    c(0.028591, -0.004629, -0.056339, 0.059056, -0.061024, -0.017879),
    c(-0.002952, 0.022709, -0.041931, 0.086825, -0.050259, 0.014297),
    c(-0.102597, -0.006659, -0.066086, 0.157154, -0.027270, 0.007630)
  )
  emb <- encode(onnx_embedder(MODEL), c("Bombay", "Mumbai", "aspirin"))
  expect_equal(emb[, 1:6], expected, tolerance = 1e-5)
})

test_that("batching does not change the result", {
  skip_onnx()
  texts <- c("Bombay", "Calcutta", "Madras", "Mumbai", "aspirin")
  one <- encode(onnx_embedder(MODEL, batch_size = 32), texts)
  many <- encode(onnx_embedder(MODEL, batch_size = 2), texts)
  expect_equal(one, many, tolerance = 1e-6)
})

test_that("padding does not leak between rows of a batch", {
  skip_onnx()
  # A long string in the batch pads the short ones. Masked pooling means the short
  # string's embedding must not move.
  alone <- encode(onnx_embedder(MODEL), "Bombay")
  padded <- encode(
    onnx_embedder(MODEL),
    c("Bombay", paste(rep("a very much longer sentence", 5), collapse = " "))
  )
  expect_equal(alone[1, ], padded[1, ], tolerance = 1e-6)
})

test_that("semantic matching beats string similarity on a known case", {
  skip_onnx()
  # "Madras"/"Chennai" share almost no characters, so string similarity cannot get it.
  refs <- c("Mumbai", "Kolkata", "Chennai")
  by_string <- alethia("Madras", refs)
  by_meaning <- alethia("Madras", refs, model = onnx_embedder(MODEL))
  expect_equal(by_string$alethia_prediction, "Mumbai")
  expect_equal(by_meaning$alethia_prediction, "Chennai")
})

test_that("use_onnx_backend resolves bare model names", {
  skip_onnx()
  old <- use_onnx_backend()
  on.exit(set_embedding_backend(old))
  out <- alethia("Bombay", c("Mumbai", "Delhi"), model = MODEL)
  expect_equal(out$alethia_prediction, "Mumbai")
  expect_equal(out$alethia_method, MODEL)
})

test_that("the embedder reports the cpu execution provider", {
  skip_onnx()
  expect_equal(as_embedder(onnx_embedder(MODEL))$family, "onnx/cpu")
})
