# Cross-language parity: every expectation is a number the Python implementation
# produced, recorded by tools/make_fixtures.py. These tests are the contract between
# the two packages, so a drift in either one fails here.

test_that("string matching reproduces Python scores on census district names", {
  fx <- load_fixture("district_string_match")
  queries <- unlist(fx$queries)
  references <- unlist(fx$references)

  got <- match_by_strings(queries, references)

  expect_equal(got$given_entity, from_json_chr(lapply(fx$expected, `[[`, "given_entity")))
  expect_equal(
    got$alethia_prediction,
    from_json_chr(lapply(fx$expected, `[[`, "alethia_prediction"))
  )
  expect_equal(
    got$alethia_score,
    from_json_num(lapply(fx$expected, `[[`, "alethia_score")),
    tolerance = TOL_STRING
  )
})

test_that("token_sort_ratio agrees with rapidfuzz to within an ULP", {
  fx <- load_fixture("district_string_match")
  queries <- unlist(fx$queries)
  references <- unlist(fx$references)
  expected <- from_json_num(lapply(fx$expected, `[[`, "alethia_score"))
  predicted <- from_json_chr(lapply(fx$expected, `[[`, "alethia_prediction"))

  # Score the exact pairs Python chose, isolating the scorer from the argmax.
  direct <- token_sort_ratio(queries, predicted) / 100
  expect_equal(direct, expected, tolerance = TOL_STRING)
})

test_that("the full pipeline agrees, including nulls and the exact prefilter", {
  fx <- load_fixture("district_pipeline")
  queries <- unlist(fx$queries)
  null_mask <- unlist(fx$queries_null_mask)
  queries[null_mask] <- NA_character_
  references <- unlist(fx$references)

  got <- alethia(queries, references, model = "rapidfuzz")

  expect_equal(
    got$alethia_prediction,
    from_json_chr(lapply(fx$expected, `[[`, "alethia_prediction"))
  )
  expect_equal(
    got$alethia_score,
    from_json_num(lapply(fx$expected, `[[`, "alethia_score")),
    tolerance = TOL_STRING
  )
})

test_that("an explicit threshold blanks the same rows in both languages", {
  fx <- load_fixture("district_threshold")
  got <- match_by_strings(
    unlist(fx$queries), unlist(fx$references),
    threshold = fx$threshold
  )
  expected_pred <- from_json_chr(lapply(fx$expected, `[[`, "alethia_prediction"))

  expect_equal(got$alethia_prediction, expected_pred)
  expect_equal(is.na(got$alethia_score), is.na(expected_pred))
  expect_true(any(is.na(expected_pred)), info = "fixture should exercise the cutoff")
})

test_that("embedding matching reproduces Python on misspelled medication names", {
  fx <- load_fixture("medication_embedding_match")
  queries <- unlist(fx$queries)
  references <- unlist(fx$references)

  got <- match_by_embeddings(queries, references, char_bag)

  expect_equal(
    got$alethia_prediction,
    from_json_chr(lapply(fx$expected, `[[`, "alethia_prediction"))
  )
  expect_equal(
    got$alethia_score,
    from_json_num(lapply(fx$expected, `[[`, "alethia_score")),
    tolerance = TOL_FLOAT32
  )
})

test_that("clustering assigns the same labels and finds the same edges", {
  fx <- load_fixture("medication_clusters")
  entities <- unlist(fx$entities)

  got <- cluster_entities(entities, char_bag, floor = fx$floor, k = fx$k)

  expect_equal(got$entities, unlist(fx$expected_entities))
  expect_equal(got$labels, unlist(fx$expected_labels))
  expect_equal(n_clusters(got), fx$expected_n_clusters)

  expected_edges <- fx$expected_edges
  expect_equal(nrow(got$edges), length(expected_edges))
  skip_if(length(expected_edges) == 0)

  # Compare the edge set keyed on the entity pair, then assert ordering separately
  # below. Keying on the pair makes a failure name which merge differs.
  py <- data.frame(
    # Python indexes from 0, R from 1.
    i = vapply(expected_edges, `[[`, numeric(1), "i") + 1,
    j = vapply(expected_edges, `[[`, numeric(1), "j") + 1,
    cosine = vapply(expected_edges, `[[`, numeric(1), "cosine"),
    confidence = vapply(expected_edges, `[[`, numeric(1), "confidence"),
    mutual = vapply(expected_edges, `[[`, logical(1), "mutual")
  )
  py <- py[order(py$i, py$j), , drop = FALSE]
  r <- got$edges[order(got$edges$i, got$edges$j), , drop = FALSE]

  expect_equal(r$i, py$i)
  expect_equal(r$j, py$j)
  expect_equal(r$cosine, py$cosine, tolerance = TOL_FLOAT32)
  expect_equal(r$confidence, py$confidence, tolerance = TOL_FLOAT32)
  expect_equal(r$mutual, py$mutual)
})

test_that("edges are ordered by decreasing confidence", {
  fx <- load_fixture("medication_clusters")
  got <- cluster_entities(unlist(fx$entities), char_bag, floor = fx$floor, k = fx$k)
  expect_false(is.unsorted(rev(got$edges$confidence)))
})

test_that("every label-free metric matches the Python value", {
  fx <- load_fixture("medication_metrics")
  refs <- unlist(fx$references)
  queries <- unlist(fx$queries)
  ref_emb <- char_bag(refs)
  query_emb <- char_bag(queries)
  expected <- fx$expected

  got <- c(
    reference_separability(ref_emb),
    centered_separability(ref_emb),
    intrinsic_dimensionality(ref_emb),
    retrieval_margin(query_emb, ref_emb),
    hubness(query_emb, ref_emb, k = 5),
    list(
      uniformity_loss = uniformity_loss(ref_emb),
      alignment_loss = alignment_loss(ref_emb, query_emb),
      mutual_nn_rate = mutual_nn_rate(query_emb, ref_emb, k = 5)
    )
  )

  # A metric Python grew and R has not ported must fail here, not pass on the
  # intersection.
  expect_setequal(names(got), names(expected))

  for (metric in names(expected)) {
    expect_equal(
      got[[metric]],
      as.numeric(expected[[metric]]),
      tolerance = TOL_FLOAT64,
      info = paste("metric:", metric)
    )
  }
})

test_that("the recommended model matches Python's", {
  fx <- load_fixture("assess_composite")
  weights <- DEFAULT_METRIC_WEIGHTS
  weights[["alignment_loss"]] <- 0
  weights[["positive_pair_rank"]] <- 0

  res <- assess_models(
    unlist(fx$queries), unlist(fx$references),
    list(char_bag = char_bag, char_bigram = char_bigram),
    weights = weights
  )

  # A user acts on the recommendation, so it is asserted exactly.
  expect_equal(res$best, fx$expected_best)

  py <- vapply(fx$expected, function(e) as.numeric(e$score), numeric(1))
  names(py) <- vapply(fx$expected, function(e) e$model, character(1))
  expect_equal(
    res$table$score,
    unname(py[res$table$model]),
    tolerance = TOL_FLOAT64
  )
})

test_that("a degenerate row is excluded from z-margins, not scored as zero", {
  # Every reference identical, so no scale exists to express a margin in. Substituting
  # 0 would drag the mean down and count as a low margin.
  flat <- matrix(1, nrow = 4, ncol = 3)
  out <- retrieval_margin(flat, flat)
  expect_true(is.nan(out$mean_margin_z) || is.na(out$mean_margin_z))
  expect_equal(out$low_margin_z_rate, 0)
})

test_that("the robustness family matches Python on the same variant pairs", {
  # Python draws its variants from CPython's Mersenne Twister, which no other language
  # reproduces, so Python generates the pairs and R scores those exact ones.
  fx <- load_fixture("robustness_pairs")
  src <- char_bag(unlist(fx$source))
  var <- char_bag(unlist(fx$variant))

  expect_equal(
    positive_pair_rank(src, var),
    as.numeric(fx$expected_positive_pair_rank),
    tolerance = TOL_FLOAT64
  )
  expect_equal(
    alignment_loss(src, var),
    as.numeric(fx$expected_alignment_loss),
    tolerance = TOL_FLOAT64
  )
})

test_that("alignment_loss is exercised on genuinely different inputs", {
  # Guard against the trivial self-comparison: alignment_loss(x, x) is 0 by
  # construction and would pass even if the function were wrong.
  fx <- load_fixture("robustness_pairs")
  src <- char_bag(unlist(fx$source))
  var <- char_bag(unlist(fx$variant))
  expect_gt(alignment_loss(src, var), 0.01)
  expect_equal(alignment_loss(src, src), 0)
})

test_that("positive_pair_rank uses mid-ranks for ties", {
  # Every row identical: each positive ties with all n variants, so the mid-rank is
  # (0 + 0.5*n)/n = 0.5. A naive strict-greater count would give 0.
  flat <- matrix(1, nrow = 4, ncol = 3)
  expect_equal(positive_pair_rank(flat, flat), 0.5)
})

test_that("positive_pair_rank rejects mismatched pair counts", {
  expect_error(
    positive_pair_rank(matrix(1, 3, 2), matrix(1, 2, 2)),
    "same number of rows"
  )
})

test_that("threshold and floor compare on quantized values, as Python does", {
  # a raw compare puts a score within floating-point noise of the cutoff on different
  # sides in different BLAS builds, so the three implementations would disagree
  near <- 0.7 - 1e-15
  expect_true(rank_key(near) >= rank_key(0.7))

  res <- data.frame(
    given_entity = "a", alethia_prediction = "b", alethia_score = near,
    stringsAsFactors = FALSE
  )
  expect_equal(apply_threshold(res, 0.7)$alethia_prediction, "b")
})
