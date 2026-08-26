// Three ways to be quietly wrong: sorting only the visible page, unmatched rows
// flooding the top when the direction flips, and no route back to the review order.
// None of them throws.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cycleSort, sortRows, sortValue } from '../src/ui/sort.js';

const row = (given, prediction, score) => ({
  given_entity: given,
  alethia_prediction: prediction,
  alethia_score: score,
});

// review order: corrected, then unmatched, then already-correct, as the view hands it
const SAMPLE = [
  row('Asprin', 'Aspirin', 0.91),
  row('Metfoormin', 'Metformin', 0.74),
  row('zzz-unknown', null, null),
  row('Ibuprofen', 'Ibuprofen', 1.0),
];

test('no sort returns the review order untouched, and the same array', () => {
  // Identity matters: the view memoises on this, and a fresh array every render would
  // re-reconcile 200 table rows for nothing.
  assert.equal(sortRows(SAMPLE, null), SAMPLE);
});

test('sorting never mutates the caller array', () => {
  const before = [...SAMPLE];
  sortRows(SAMPLE, { key: 'score', dir: 'desc' });
  assert.deepEqual(SAMPLE, before, 'the review order must survive a sort for the reset');
});

test('score sorts numerically in both directions', () => {
  const asc = sortRows(SAMPLE, { key: 'score', dir: 'asc' }).map((r) => r.alethia_score);
  assert.deepEqual(asc, [0.74, 0.91, 1.0, null]);

  const desc = sortRows(SAMPLE, { key: 'score', dir: 'desc' }).map((r) => r.alethia_score);
  assert.deepEqual(desc, [1.0, 0.91, 0.74, null]);
});

test('unmatched rows stay last in both directions', () => {
  // The bug this catches: negating the comparator sends nulls to the top on a descending
  // sort, so "highest score first" opens with a screen of blanks.
  for (const key of ['prediction', 'score']) {
    for (const dir of ['asc', 'desc']) {
      const sorted = sortRows(SAMPLE, { key, dir });
      assert.equal(
        sortValue(sorted.at(-1), key), null,
        `${key} ${dir} should leave the unmatched row at the bottom`,
      );
    }
  }
});

test('text columns sort by locale, not by code point', () => {
  const accented = [row('Zurich', 'Zurich', 1), row('Ätna', 'Atna', 1), row('apple', 'apple', 1)];
  const given = sortRows(accented, { key: 'given', dir: 'asc' }).map((r) => r.given_entity);
  // A raw < comparison puts every capital ahead of every lowercase letter, which reads as
  // scrambled to anyone who is not thinking in ASCII.
  assert.deepEqual(given, ['apple', 'Ätna', 'Zurich']);
});

test('sorting orders the whole set, so the preview shows the true extremes', () => {
  // The regression this guards: sorting after the 200-row slice. With 300 rows the lowest
  // score lives outside the preview, and a naive implementation would never surface it.
  const many = Array.from({ length: 300 }, (_, i) => row(`e${i}`, `r${i}`, (i + 1) / 1000));
  const worst = sortRows(many, { key: 'score', dir: 'asc' })[0];
  assert.equal(worst.alethia_score, 0.001);
  const best = sortRows(many, { key: 'score', dir: 'desc' })[0];
  assert.equal(best.alethia_score, 0.3);
});

test('a column cycles ascending, descending, then back to review order', () => {
  let s = null;
  s = cycleSort(s, 'score');
  assert.deepEqual(s, { key: 'score', dir: 'asc' });
  s = cycleSort(s, 'score');
  assert.deepEqual(s, { key: 'score', dir: 'desc' });
  s = cycleSort(s, 'score');
  assert.equal(s, null, 'the third activation must return to review order');
});

test('switching columns starts the new one ascending', () => {
  const s = cycleSort({ key: 'score', dir: 'desc' }, 'given');
  assert.deepEqual(s, { key: 'given', dir: 'asc' },
    'inheriting the previous direction makes the first click on a column unpredictable');
});

test('sortValue treats an empty prediction as no match, not as an empty string', () => {
  // The worker reports no-match as null, but a defensive '' would sort to the top of an
  // ascending text sort and read as a row that matched something blank.
  assert.equal(sortValue(row('x', '', null), 'prediction'), null);
  assert.equal(sortValue(row('x', null, null), 'prediction'), null);
  assert.equal(sortValue(row('x', 'y', 0), 'score'), 0, 'a genuine zero is not absence');
});
