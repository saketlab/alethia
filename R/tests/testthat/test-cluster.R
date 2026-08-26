test_that("spelling variants merge and unrelated strings stay apart", {
  ents <- c("color", "colour", "flavour", "flavor", "zzzzzzzzzzzz")
  res <- cluster_entities(ents, char_bag, floor = 0.8, k = 3)
  groups <- split(res$entities, res$labels)
  expect_true(any(vapply(groups, function(g) all(c("color", "colour") %in% g), logical(1))))
  expect_true(any(vapply(groups, function(g) identical(g, "zzzzzzzzzzzz"), logical(1))))
})

test_that("entities are de-duplicated on exact text first", {
  res <- cluster_entities(c("a", "a", "b"), char_bag, floor = 0.1)
  expect_equal(res$entities, c("a", "b"))
})

test_that("a content-blind embedder merges nothing confidently", {
  ents <- c("color", "colour", "flavour", "flavor")
  blind <- function(texts) matrix(rep(c(1, 0), length(texts)), nrow = length(texts), byrow = TRUE)
  # Every vector identical: cosine is 1 everywhere, so margins are 0 and there is no
  # evidence favouring any particular merge over any other.
  res <- cluster_entities(ents, blind, floor = 0.99, k = 3)
  expect_equal(nrow(res$edges), choose(length(ents), 2))
})

test_that("a high floor produces no edges at all", {
  res <- cluster_entities(c("color", "zzzzzzzzz"), char_bag, floor = 0.999)
  expect_equal(nrow(res$edges), 0)
  expect_equal(n_clusters(res), 2)
})

test_that("min_confidence drops weak edges", {
  ents <- c("color", "colour", "zzzzzzzzz")
  keep <- cluster_entities(ents, char_bag, floor = 0.5)
  drop <- cluster_entities(ents, char_bag, floor = 0.5, min_confidence = 99)
  expect_lt(n_clusters(keep), n_clusters(drop))
})

test_that("canonical names pick the shortest member by default", {
  res <- cluster_entities(c("color", "colouring"), char_bag, floor = 0.5, k = 1)
  recs <- cluster_records(res)
  expect_true(all(nchar(recs$canonical) <= nchar(recs$entity)))
})

test_that("non-finite embeddings are rejected", {
  bad <- function(texts) {
    m <- char_bag(texts)
    m[1, 1] <- NaN
    m
  }
  expect_error(cluster_entities(c("a", "b"), bad), "non-finite|Non-finite")
})

test_that("fewer than two entities yields no edges", {
  res <- cluster_entities("only", char_bag)
  expect_equal(nrow(res$edges), 0)
  expect_equal(n_clusters(res), 1)
})

test_that("cluster_records returns one row per entity", {
  res <- cluster_entities(c("color", "colour", "zzz"), char_bag, floor = 0.8)
  recs <- cluster_records(res)
  expect_equal(nrow(recs), 3)
  expect_setequal(names(recs), c("entity", "cluster", "canonical"))
})
