test_that("token_sort_ratio is order-insensitive but case-sensitive", {
  expect_equal(token_sort_ratio("New Delhi", "Delhi New"), 100)
  expect_equal(token_sort_ratio("Bombay", "Bombay"), 100)
  # rapidfuzz does not lowercase, and neither do we.
  expect_lt(token_sort_ratio("bombay", "Bombay"), 100)
})

test_that("two empty strings are identical rather than 0/0", {
  expect_equal(token_sort_ratio("", ""), 100)
  expect_equal(token_sort_ratio("a", ""), 0)
})

test_that("token_sort collapses whitespace and sorts", {
  expect_equal(token_sort("  b   a "), "a b")
  expect_equal(token_sort(""), "")
})

test_that("exact matches short-circuit and score 1", {
  out <- alethia(c("Mumbai"), c("Mumbai", "Delhi"))
  expect_equal(out$alethia_prediction, "Mumbai")
  expect_equal(out$alethia_score, 1)
  expect_equal(out$alethia_method, "exact")
})

test_that("the exact prefilter is case-insensitive by default", {
  out <- alethia("mumbai", c("Mumbai", "Delhi"))
  expect_equal(out$alethia_method, "exact")
  out <- alethia("mumbai", c("Mumbai", "Delhi"), exact_match_case_sensitive = TRUE)
  expect_equal(out$alethia_method, "rapidfuzz")
})

test_that("null-like entries are preserved as no-match rows", {
  out <- alethia(
    c("Bombay", NA, "N/A", "nan", ""), c("Mumbai"),
    drop_duplicates = FALSE
  )
  expect_equal(nrow(out), 5)
  expect_equal(out$alethia_method, c("rapidfuzz", "nan", "nan", "nan", "nan"))
  expect_true(all(is.na(out$alethia_prediction[-1])))
})

test_that("an empty reference list yields no match, not a self-match", {
  out <- alethia(c("anything"), character(0))
  expect_true(is.na(out$alethia_prediction))
  expect_true(is.na(out$alethia_score))
})

test_that("drop_duplicates matches the Python default but can be turned off", {
  q <- c("Bombay", "Bombay", "Calcutta")
  expect_equal(nrow(alethia(q, c("Mumbai", "Kolkata"))), 2)
  expect_equal(
    nrow(alethia(q, c("Mumbai", "Kolkata"), drop_duplicates = FALSE)),
    3
  )
})

test_that("a threshold blanks both the prediction and the score", {
  out <- match_by_strings("zzzzzzzz", c("Mumbai", "Delhi"), threshold = 0.95)
  expect_true(is.na(out$alethia_prediction))
  expect_true(is.na(out$alethia_score))
})

test_that("no threshold returns the best guess whatever it scores", {
  out <- match_by_strings("zzzzzzzz", c("Mumbai", "Delhi"))
  expect_false(is.na(out$alethia_prediction))
})

test_that("empty input is handled without error", {
  expect_equal(nrow(match_by_strings(character(0), c("a"))), 0)
  expect_equal(nrow(alethia(character(0), c("a"))), 0)
})
