// Cross-language parity: the same JSON fixtures the R package reads, generated from
// Python. A drift in any of the three fails here.
//
// The embedder is char_bag, identical in all three languages; a real transformer would
// test ONNX Runtime Web.

import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  clusterEntities,
  fromRows,
  l2Normalize,
  matchByEmbeddings,
  mutualNnEdges,
  rankKey,
} from '../src/core.js';
import {
  alignmentLoss,
  centeredSeparability,
  hubness,
  intrinsicDimensionality,
  mutualNnRate,
  positivePairRank,
  referenceSeparability,
  retrievalMargin,
  uniformityLoss,
} from '../src/metrics.js';
import { compositeScores } from '../src/assess.js';

// The R package owns the fixtures; this reads the same files rather than a second copy.
// A checkout without R/ skips these instead of failing, since the browser build does not
// depend on it.
const FIXTURES = process.env.ALETHIA_FIXTURES
  ?? new URL('../../R/tests/testthat/fixtures/', import.meta.url).pathname;

const haveFixtures = existsSync(join(FIXTURES, 'district_string_match.json'));

const load = (name) => JSON.parse(readFileSync(join(FIXTURES, `${name}.json`), 'utf8'));

const VOCAB = 'abcdefghijklmnopqrstuvwxyz ';
function charBag(texts) {
  const rows = texts.map((text) => {
    const v = new Float64Array(VOCAB.length);
    for (const ch of String(text).toLowerCase()) {
      const pos = VOCAB.indexOf(ch);
      if (pos >= 0) v[pos] += 1;
    }
    return v;
  });
  return fromRows(rows);
}

// Tolerances by path, matching the R harness. Both sides are float64 on identical
// input, so the only slack is accumulation order in the matrix products.
const TOL = 1e-9;

function closeTo(actual, expected, tol, label) {
  if (expected === null || Number.isNaN(expected)) {
    assert.ok(
      actual === null || Number.isNaN(actual),
      `${label}: expected null/NaN, got ${actual}`,
    );
    return;
  }
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${label}: got ${actual}, expected ${expected} (diff ${Math.abs(actual - expected)})`,
  );
}

test('embedding matching reproduces Python predictions and scores', { skip: !haveFixtures }, () => {
  const fx = load('medication_embedding_match');
  const got = matchByEmbeddings(
    fx.queries, fx.references, charBag(fx.queries), charBag(fx.references),
  );
  assert.equal(got.length, fx.expected.length);
  got.forEach((row, i) => {
    assert.equal(row.alethia_prediction, fx.expected[i].alethia_prediction,
      `row ${i} prediction`);
    closeTo(row.alethia_score, fx.expected[i].alethia_score, TOL, `row ${i} score`);
  });
});

test('clustering assigns the same labels and edges', { skip: !haveFixtures }, () => {
  const fx = load('medication_clusters');
  const unique = [...new Set(fx.entities)];
  const res = clusterEntities(fx.entities, charBag(unique), {
    floor: fx.floor, k: fx.k,
  });

  assert.deepEqual(res.entities, fx.expected_entities);
  assert.deepEqual(res.labels, fx.expected_labels);
  assert.equal(new Set(res.labels).size, fx.expected_n_clusters);

  // Compare the edge set keyed on the entity pair, then ordering separately, so a
  // failure names which merge differs, not which position.
  const key = (e) => `${e.i}-${e.j}`;
  const py = new Map(fx.expected_edges.map((e) => [key(e), e]));
  assert.equal(res.edges.length, fx.expected_edges.length, 'edge count');
  for (const edge of res.edges) {
    const want = py.get(key(edge));
    assert.ok(want, `edge ${key(edge)} not in the Python edge set`);
    closeTo(edge.cosine, want.cosine, TOL, `edge ${key(edge)} cosine`);
    closeTo(edge.confidence, want.confidence, TOL, `edge ${key(edge)} confidence`);
    assert.equal(edge.mutual, want.mutual, `edge ${key(edge)} mutual`);
  }
});

test('every label-free metric matches the Python value', { skip: !haveFixtures }, () => {
  const fx = load('medication_metrics');
  const refEmb = charBag(fx.references);
  const queryEmb = charBag(fx.queries);

  const got = {
    ...referenceSeparability(refEmb),
    ...centeredSeparability(refEmb),
    ...intrinsicDimensionality(refEmb),
    ...retrievalMargin(queryEmb, refEmb),
    ...hubness(queryEmb, refEmb, 5),
    uniformity_loss: uniformityLoss(refEmb),
    alignment_loss: alignmentLoss(refEmb, queryEmb),
    mutual_nn_rate: mutualNnRate(queryEmb, refEmb, 5),
  };

  // A metric Python grew and the browser has not ported must fail here, not pass on
  // silently checking only the intersection.
  assert.deepEqual(
    Object.keys(got).sort(), Object.keys(fx.expected).sort(),
    'metric names differ from the Python fixture',
  );
  for (const [name, expected] of Object.entries(fx.expected)) {
    closeTo(got[name], expected, TOL, name);
  }
});

test('the robustness family matches Python on the same variant pairs', { skip: !haveFixtures }, () => {
  // Python draws the variants from CPython's Mersenne
  // Twister, which no other language reproduces. Python supplies the pairs; this scores
  // them, which tests the metric and not the RNG.
  const fx = load('robustness_pairs');
  const src = charBag(fx.source);
  const variant = charBag(fx.variant);
  closeTo(positivePairRank(src, variant), fx.expected_positive_pair_rank, TOL,
    'positive_pair_rank');
  closeTo(alignmentLoss(src, variant), fx.expected_alignment_loss, TOL,
    'alignment_loss');
});

test('l2Normalize leaves zero rows alone rather than producing NaN', { skip: !haveFixtures }, () => {
  const out = l2Normalize(fromRows([[3, 4], [0, 0]]));
  closeTo(out.data[0], 0.6, 1e-15, 'row 0 x');
  closeTo(out.data[1], 0.8, 1e-15, 'row 0 y');
  assert.ok(out.data.every(Number.isFinite), 'zero row produced a non-finite value');
});

test('a degenerate row is excluded from z-margins, not scored as zero', { skip: !haveFixtures }, () => {
  const flat = fromRows([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]);
  const out = retrievalMargin(flat, flat);
  assert.ok(Number.isNaN(out.mean_margin_z), 'mean_margin_z should be NaN');
  assert.equal(out.low_margin_z_rate, 0, 'undefined z is not a low margin');
});

test('a single reference yields the degenerate constants, matching Python', { skip: !haveFixtures }, () => {
  const out = retrievalMargin(fromRows([[1, 0], [0, 1]]), fromRows([[1, 0]]));
  assert.deepEqual(out, {
    mean_margin: 0, low_margin_rate: 1, mean_margin_z: 0, low_margin_z_rate: 1,
  });
});

test('positivePairRank uses mid-ranks for ties', { skip: !haveFixtures }, () => {
  const flat = fromRows([[1, 1], [1, 1], [1, 1], [1, 1]]);
  assert.equal(positivePairRank(flat, flat), 0.5);
});

test('mutualNnEdges is ordered by decreasing confidence', { skip: !haveFixtures }, () => {
  const fx = load('medication_clusters');
  const unique = [...new Set(fx.entities)];
  const edges = mutualNnEdges(charBag(unique), { floor: fx.floor, k: fx.k });
  for (let i = 1; i < edges.length; i++) {
    assert.ok(edges[i - 1].confidence >= edges[i].confidence, `edge ${i} out of order`);
  }
});

test('the composite ranking and winner match Python', { skip: !haveFixtures }, () => {
  // A user acts on the recommendation, so it is asserted exactly.
  const fx = load('assess_composite');
  const BIGRAM_DIMS = 64;
  const charBigram = (texts) => fromRows(texts.map((text) => {
    const v = new Float64Array(BIGRAM_DIMS);
    const s = String(text).toLowerCase();
    for (let i = 0; i < s.length - 1; i++) {
      const a = VOCAB.indexOf(s[i]);
      const b = VOCAB.indexOf(s[i + 1]);
      if (a >= 0 && b >= 0) v[(a * 31 + b) % BIGRAM_DIMS] += 1;
    }
    return v;
  }));

  const embedders = { char_bag: charBag, char_bigram: charBigram };
  const assessments = Object.entries(embedders).map(([model, fn]) => {
    const refEmb = fn(fx.references);
    const queryEmb = fn(fx.queries);
    return {
      model,
      error: null,
      metrics: {
        ...referenceSeparability(refEmb),
        ...centeredSeparability(refEmb),
        ...intrinsicDimensionality(refEmb),
        ...retrievalMargin(queryEmb, refEmb),
        ...hubness(queryEmb, refEmb, 5),
        uniformity_loss: uniformityLoss(refEmb),
        mutual_nn_rate: mutualNnRate(queryEmb, refEmb, 5),
      },
    };
  });

  const scored = compositeScores(assessments);
  assert.equal(scored[0].model, fx.expected_best, 'recommended model');
  const byModel = Object.fromEntries(scored.map((s) => [s.model, s.score]));
  for (const row of fx.expected) {
    closeTo(byModel[row.model], row.score, TOL, `composite score for ${row.model}`);
  }
});

test('threshold and floor compare on quantized values, as Python and R do', { skip: !haveFixtures }, () => {
  // a raw compare puts a score within floating-point noise of the cutoff on different
  // sides in different builds, so the three implementations would disagree
  const near = 0.7 - 1e-15;
  assert.ok(rankKey(near) >= rankKey(0.7));

  const [row] = matchByEmbeddings(['q'], ['r'],
    fromRows([[1, 0]]), fromRows([[1, 0]]), 1);
  assert.equal(row.alethia_prediction, 'r', 'an exact match must clear its own cutoff');
});
