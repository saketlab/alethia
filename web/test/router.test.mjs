// Routing makes a run shareable, so it is asserted here.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildHash, parseHash } from '../src/ui/router.js';

test('an empty hash lands on the match view', () => {
  assert.deepEqual(parseHash(''), { view: 'match', params: {} });
  assert.deepEqual(parseHash('#'), { view: 'match', params: {} });
  assert.deepEqual(parseHash('#/'), { view: 'match', params: {} });
});

test('a shared link carries the model through', () => {
  const url = buildHash('match', { model: 'Xenova/bge-small-en-v1.5' });
  assert.equal(url, '#/match?model=Xenova%2Fbge-small-en-v1.5');
  const route = parseHash(url);
  assert.equal(route.view, 'match');
  assert.equal(route.params.model, 'Xenova/bge-small-en-v1.5');
});

test('the choose view is addressable', () => {
  assert.deepEqual(parseHash('#/choose'), { view: 'choose', params: {} });
  assert.equal(buildHash('choose'), '#/choose');
});

test('an unknown view falls back rather than rendering nothing', () => {
  assert.equal(parseHash('#/nonsense').view, 'match');
});

test('empty params are omitted so links stay clean', () => {
  assert.equal(buildHash('match', { model: null, threshold: '' }), '#/match');
});

test('a model id containing slashes survives a round trip', () => {
  // Hugging Face ids always contain a slash; a naive split would truncate them.
  const id = 'sentence-transformers/all-MiniLM-L6-v2';
  assert.equal(parseHash(buildHash('match', { model: id })).params.model, id);
});
