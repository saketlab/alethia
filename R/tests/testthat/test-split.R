test_that("picks the best scoring piece", {
  out <- alethia_split(c("apple, zzzzzzzzzz"), c("apple", "banana"), split_on = ",")
  expect_equal(out$alethia_prediction, "apple")
  expect_equal(out$alethia_matched_part, "apple")
})

test_that("provenance keeps every piece", {
  out <- alethia_split(c("apple, banana"), c("apple", "banana"), split_on = ",")
  parts <- out$alethia_parts[[1]]
  got <- vapply(parts, function(p) p$part, character(1))
  expect_setequal(got, c("apple", "banana"))
  expect_gte(parts[[1]]$score, parts[[2]]$score)
})

test_that("no separator matches the whole entry", {
  out <- alethia_split(c("apple"), c("apple", "banana"), split_on = ",")
  expect_equal(out$alethia_matched_part, "apple")
  expect_equal(out$alethia_prediction, "apple")
})

test_that("null-like entries pass through", {
  out <- alethia_split(c(NA), c("apple"), split_on = ",")
  expect_true(is.na(out$alethia_prediction))
  expect_true(is.na(out$alethia_score))
})

test_that("repeated parts across entries stay aligned", {
  # the deduped "aple" result has to be looked up by every entry that used it
  out <- alethia_split(
    c("aple, aple", "aple, bananna"),
    c("apple", "banana"),
    split_on = ","
  )
  expect_equal(out$alethia_prediction, c("apple", "banana"))
  expect_equal(out$alethia_matched_part[2], "bananna")
})

test_that("a multiword entry without the separator is not split on words", {
  out <- alethia_split(
    c("Cns Tuberculosis"),
    c("Tuberculosis of nervous system"),
    split_on = ","
  )
  expect_equal(length(out$alethia_parts[[1]]), 1)
  expect_equal(out$alethia_matched_part, "Cns Tuberculosis")
})
