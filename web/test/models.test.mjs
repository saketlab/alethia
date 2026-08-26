// The model roster is data, and every field in it is load-bearing somewhere the type
// system cannot see: pooling reaches transformers.js through a postMessage, size is
// parsed back into a number to total the download, and id has to name a repo publishing
// ONNX weights. A typo in any of them fails at runtime, on the user's machine.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { DEFAULT_MODEL_ID, MODELS, resolveModel } from '../src/ui/models.js';

test('every model declares the fields the UI and worker read', () => {
  for (const m of MODELS) {
    assert.match(m.id, /^[\w-]+\/[\w.-]+$/, `${m.id} is not a Hugging Face repo id`);
    assert.ok(m.label, `${m.id} needs a label`);
    assert.ok(m.note, `${m.id} needs a note -- it is the only guidance on when to pick it`);
    assert.equal(typeof m.default, 'boolean', `${m.id} must state whether it is pre-ticked`);

    // ChooseView totals the download with parseInt on this string.
    assert.match(m.size, /^\d+ MB$/, `${m.id} size must parse as a number of MB`);
    assert.ok(Number.isFinite(parseInt(m.size, 10)));
  }
});

test('ids and labels are unique', () => {
  // Labels are the join key between the worker's result and the model list: the assess
  // job reports `best` by label and the UI maps it back to an id. Two models sharing a
  // label would silently hand the user the wrong winner.
  assert.equal(new Set(MODELS.map((m) => m.id)).size, MODELS.length, 'duplicate id');
  assert.equal(new Set(MODELS.map((m) => m.label)).size, MODELS.length, 'duplicate label');
});

test('pooling is either omitted or a strategy transformers.js implements', () => {
  // An unrecognised value throws inside the pipeline, mid-run, after the download.
  for (const m of MODELS) {
    if (m.pooling === undefined) continue;
    assert.ok(['mean', 'cls'].includes(m.pooling), `${m.id} has pooling "${m.pooling}"`);
  }
});

test('SapBERT is CLS-pooled, as its model card requires', () => {
  // Regression guard with teeth: mean-pooling SapBERT does not throw and does not look
  // wrong. It measurably ranks worse on the biomedical lists it exists to handle, which
  // is exactly the kind of defect that survives a code review.
  const sapbert = MODELS.find((m) => m.id.includes('SapBERT'));
  assert.ok(sapbert, 'SapBERT should be offered');
  assert.equal(sapbert.pooling, 'cls');
});

test('the sentence-transformers models stay on mean pooling', () => {
  // Their agreement with the Python and R packages depends on it.
  for (const m of MODELS.filter((x) => !x.id.includes('SapBERT'))) {
    assert.ok(m.pooling === undefined || m.pooling === 'mean', `${m.id} changed pooling`);
  }
});

test('the default model exists and resolveModel falls back to it', () => {
  assert.ok(MODELS.some((m) => m.id === DEFAULT_MODEL_ID), 'DEFAULT_MODEL_ID must be real');
  assert.equal(resolveModel(DEFAULT_MODEL_ID).id, DEFAULT_MODEL_ID);
  assert.equal(resolveModel('nonexistent/model').id, DEFAULT_MODEL_ID);
  assert.equal(resolveModel(undefined).id, DEFAULT_MODEL_ID);
});

test('at least two models are pre-ticked, and their total download stays modest', () => {
  // Two models minimum to rank anything, and a ceiling so a first run does not download
  // a quarter of a gigabyte. 120 MB leaves no room for mpnet (109 MB) or GTE-large
  // (336 MB) to drift in beside the 57 MB defaults.
  const ticked = MODELS.filter((m) => m.default);
  assert.ok(ticked.length >= 2, 'a single model has nothing to be ranked against');
  const mb = ticked.reduce((sum, m) => sum + parseInt(m.size, 10), 0);
  assert.ok(mb <= 120, `pre-ticked models total ${mb} MB`);
});

test('the pre-ticked pair is the accuracy-per-byte choice, not the biggest', () => {
  // mpnet-base loses to GTE-small on ICD-10 by 4.2 points at three times the size,
  // which no code check could catch. This pins the measurement.
  const ticked = MODELS.filter((m) => m.default).map((m) => m.label);
  assert.deepEqual(ticked.sort(), ['GTE-small', 'MiniLM-L6']);
});
