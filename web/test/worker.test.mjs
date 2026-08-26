// The worker runs in its own scope, so a name it cannot reach fails only at runtime.
// Nothing else in the suite loads it: the app harness stubs it out and the numeric tests
// import core.js directly, which is how a ReferenceError in prepare() reached the browser.

import assert from 'node:assert/strict';
import { mkdirSync, rmSync } from 'node:fs';
import { after, before, test } from 'node:test';

import { build } from 'vite';

const OUT = './.wtest';
const ENTRY = new URL('../.wtest/worker.js', import.meta.url).href;
let posted = [];

before(async () => {
  mkdirSync(OUT, { recursive: true });
  await build({
    logLevel: 'silent',
    build: {
      outDir: OUT,
      emptyOutDir: true,
      ssr: 'src/worker.js',
      rollupOptions: { external: ['@huggingface/transformers'] },
    },
  });
  globalThis.self = {
    postMessage: (m) => posted.push(m),
    set onmessage(fn) { this._h = fn; },
    get onmessage() { return this._h; },
  };
  await import(ENTRY);
});

after(() => rmSync(OUT, { recursive: true, force: true }));

const drive = async (message) => {
  posted = [];
  await self.onmessage({ data: message });
  return posted;
};

const noReferenceError = (msgs) => {
  const err = msgs.find((m) => m.type === 'error');
  assert.ok(
    !err || !/is not defined|undefined is not/.test(err.payload.message),
    `worker scope broken: ${err?.payload?.message}`,
  );
};

const MODEL = { id: 'stub/none', label: 'Stub', pooling: 'mean' };

test('match reaches the download stage with every name in scope', async () => {
  const msgs = await drive({
    type: 'match',
    payload: { queries: ['a'], references: ['a', 'b'], model: MODEL, threshold: null },
  });
  noReferenceError(msgs);
  assert.ok(msgs.some((m) => m.type === 'stage'), 'prepare() must report its first stage');
});

test('cluster reaches the download stage', async () => {
  const msgs = await drive({
    type: 'cluster',
    payload: { entities: ['a', 'b'], model: MODEL, floor: 0.8 },
  });
  noReferenceError(msgs);
  assert.ok(msgs.some((m) => m.type === 'stage'));
});

test('assess reaches the download stage', async () => {
  const msgs = await drive({
    type: 'assess',
    payload: { queries: ['a'], references: ['a', 'b'], models: [MODEL] },
  });
  noReferenceError(msgs);
  assert.ok(msgs.some((m) => m.type === 'stage'));
});

test('an unreachable model surfaces as an error message, not a crash', async () => {
  const msgs = await drive({
    type: 'match',
    payload: { queries: ['a'], references: ['b'], model: MODEL, threshold: null },
  });
  assert.ok(msgs.some((m) => m.type === 'error'), 'a failed load must reach the UI');
});
