import assert from 'node:assert/strict';
import { test } from 'node:test';

import { barFractions } from '../src/ui/chart-scale.js';

test('higher-is-better puts the largest value at the longest bar', () => {
  const [a, b, c] = barFractions([0.6, 0.4, 0.1]);
  assert.ok(a > b && b > c);
  assert.equal(a, 1);
});

test('lower-is-better flips, so the winner is not drawn as the loser', () => {
  // hubness skew: 0.9 is the best of the three and must draw longest
  const [a, b, c] = barFractions([0.9, 1.6, 2.2], { better: 'lower' });
  assert.ok(a > b && b > c, 'a lower value must produce a longer bar');
  assert.equal(a, 1);
});

test('the weakest model still gets a visible bar', () => {
  const f = barFractions([1, 0]);
  assert.ok(f[1] >= 0.08, 'a zero-length bar reads as missing data, not as last place');
});

test('a constant metric gives every model the same bar', () => {
  const f = barFractions([0.5, 0.5, 0.5]);
  assert.equal(new Set(f).size, 1);
});

test('non-finite values drop out rather than throwing', () => {
  const f = barFractions([1, NaN, 0]);
  assert.equal(f[1], 0);
  assert.ok(Number.isFinite(f[0]) && Number.isFinite(f[2]));
});
