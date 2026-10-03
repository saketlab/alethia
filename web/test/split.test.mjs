// Splitting composite entries before matching: same idea as alethia_split in Python/R.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  bestOfParts, regroupSplitMatches, splitAndDedupe, splitEntry,
} from '../src/core.js';

test('splits on the literal separator and trims pieces', () => {
  assert.deepEqual(splitEntry('CNS Tuberculosis, Septic Shock', ','), [
    'CNS Tuberculosis', 'Septic Shock',
  ]);
});

test('falls back to the whole entry when nothing usable survives', () => {
  assert.deepEqual(splitEntry('no separator here', ','), ['no separator here']);
  assert.deepEqual(splitEntry(' , , ', ','), [' , , ']);
});

test('a multiword entry without the separator is not split on words', () => {
  assert.deepEqual(splitEntry('Cns Tuberculosis', ','), ['Cns Tuberculosis']);
});

test('bestOfParts picks the highest-scoring piece and ties break first', () => {
  const matches = [
    { given_entity: 'a', alethia_prediction: 'A', alethia_score: 0.5 },
    { given_entity: 'b', alethia_prediction: 'B', alethia_score: 0.9 },
    { given_entity: 'c', alethia_prediction: 'C', alethia_score: 0.9 },
  ];
  const best = bestOfParts(matches);
  assert.equal(best.matchedPart, 'b');
  assert.equal(best.prediction, 'B');
  assert.equal(best.score, 0.9);
  assert.equal(best.parts[0].part, 'b');
  assert.equal(best.parts.length, 3);
});

test('a null match still ranks, just last', () => {
  const matches = [
    { given_entity: 'a', alethia_prediction: null, alethia_score: null },
    { given_entity: 'b', alethia_prediction: 'B', alethia_score: 0.4 },
  ];
  const best = bestOfParts(matches);
  assert.equal(best.matchedPart, 'b');
});

test('splitAndDedupe gives one code per piece, one entry per unique piece', () => {
  const { partsPerQuery, uniqueParts, codes } = splitAndDedupe(
    ['aple, aple', 'aple, bananna'], ',',
  );
  assert.deepEqual(partsPerQuery, [['aple', 'aple'], ['aple', 'bananna']]);
  assert.deepEqual(uniqueParts, ['aple', 'bananna']);
  assert.deepEqual(codes, [0, 0, 0, 1]);
});

test('regroupSplitMatches slices a repeated piece back to every query that used it', () => {
  // the same deduped match for "aple" has to serve both of its occurrences
  const queries = ['aple, aple', 'aple, bananna'];
  const partsPerQuery = [['aple', 'aple'], ['aple', 'bananna']];
  const codes = [0, 0, 0, 1];
  const uniqueMatches = [
    { given_entity: 'aple', alethia_prediction: 'apple', alethia_score: 0.9 },
    { given_entity: 'bananna', alethia_prediction: 'banana', alethia_score: 0.95 },
  ];

  const matches = regroupSplitMatches(queries, partsPerQuery, codes, uniqueMatches);
  assert.equal(matches[0].alethia_prediction, 'apple');
  assert.equal(matches[0].alethia_parts.length, 2);
  assert.equal(matches[1].alethia_prediction, 'banana');
  assert.equal(matches[1].alethia_matched_part, 'bananna');
});
