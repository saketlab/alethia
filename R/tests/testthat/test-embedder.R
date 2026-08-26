test_that("l2_normalize leaves zero rows alone rather than producing NaN", {
  x <- rbind(c(3, 4), c(0, 0))
  out <- l2_normalize(x)
  expect_equal(out[1, ], c(0.6, 0.8))
  expect_equal(out[2, ], c(0, 0))
  expect_true(all(is.finite(out)))
})

test_that("a function is accepted as an embedder", {
  e <- as_embedder(char_bag, name = "bag")
  expect_s3_class(e, "alethia_embedder")
  expect_equal(e$name, "bag")
  expect_equal(nrow(encode(e, c("a", "b", "c"))), 3)
})

test_that("an embedder passes through as_embedder unchanged", {
  e <- as_embedder(char_bag)
  expect_identical(as_embedder(e), e)
})

test_that("string-similarity keywords are rejected as embedders", {
  expect_error(as_embedder("rapidfuzz"), "not an embedding model")
  expect_error(as_embedder("exact"), "not an embedding model")
})

test_that("a model name without a registered backend fails with guidance", {
  old <- set_embedding_backend(NULL)
  on.exit(set_embedding_backend(old))
  expect_error(as_embedder("all-MiniLM-L6-v2"), "set_embedding_backend")
})

test_that("a registered backend resolves model names", {
  old <- set_embedding_backend(function(name) char_bag)
  on.exit(set_embedding_backend(old))
  e <- as_embedder("any-model")
  expect_equal(e$name, "any-model")
  expect_equal(nrow(encode(e, c("x", "y"))), 2)
})

test_that("an embedder returning the wrong number of rows is rejected", {
  bad <- as_embedder(function(texts) matrix(0, nrow = 1, ncol = 3))
  expect_error(encode(bad, c("a", "b")), "returned 1 rows for 2 inputs")
})

test_that("non-finite embeddings fail loudly", {
  bad <- as_embedder(function(texts) matrix(NaN, nrow = length(texts), ncol = 2))
  expect_error(encode(bad, c("a", "b")), "non-finite")
})

test_that("cosine matching finds the nearest reference", {
  out <- match_by_embeddings(c("colour"), c("color", "zzzzzzzzz"), char_bag)
  expect_equal(out$alethia_prediction, "color")
  expect_gt(out$alethia_score, 0.9)
})

test_that("a threshold above the best score yields no match", {
  out <- match_by_embeddings("colour", c("zzzzzzzzz"), char_bag, threshold = 0.99)
  expect_true(is.na(out$alethia_prediction))
})
