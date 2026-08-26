// The colour scheme is a preference people notice immediately when it is wrong, so the
// resolution logic is asserted, not eyeballed in one lighting condition.
import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';

let cleanup;
let React;
let act;
let createRoot;
let useAppearance;

function setSystemDark(isDark) {
  const listeners = new Set();
  window.matchMedia = (query) => ({
    media: query,
    matches: query.includes('dark') ? isDark : !isDark,
    addEventListener: (_, fn) => listeners.add(fn),
    removeEventListener: (_, fn) => listeners.delete(fn),
  });
  return (next) => listeners.forEach((fn) => fn({ matches: next }));
}

/** Render the hook and expose its latest value. */
async function renderHook() {
  const seen = {};
  function Probe() {
    Object.assign(seen, useAppearance());
    return null;
  }
  const container = document.createElement('div');
  const root = createRoot(container);
  await act(async () => { root.render(React.createElement(Probe)); });
  return { seen, rerender: async (fn) => { await act(async () => fn()); } };
}

before(async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const jsdom = (await import('global-jsdom')).default;
  cleanup = jsdom('', { url: 'http://localhost/' });
  React = await import('react');
  ({ act } = React);
  ({ createRoot } = await import('react-dom/client'));
  ({ useAppearance } = await import('../src/ui/useAppearance.js'));
});

after(() => cleanup?.());
beforeEach(() => window.localStorage.clear());

test('defaults to following the system, in dark', async () => {
  setSystemDark(true);
  const { seen } = await renderHook();
  assert.equal(seen.preference, 'system');
  assert.equal(seen.appearance, 'dark');
});

test('defaults to following the system, in light', async () => {
  setSystemDark(false);
  const { seen } = await renderHook();
  assert.equal(seen.preference, 'system');
  assert.equal(seen.appearance, 'light');
});

test('an explicit choice overrides the system and is remembered', async () => {
  setSystemDark(true);
  const { seen, rerender } = await renderHook();
  await rerender(() => seen.choose('light'));
  assert.equal(seen.appearance, 'light', 'explicit light should win over a dark system');
  assert.equal(window.localStorage.getItem('alethia.appearance'), 'light');

  // A fresh mount must honour what was stored.
  const second = await renderHook();
  assert.equal(second.seen.preference, 'light');
  assert.equal(second.seen.appearance, 'light');
});

test('while following the system it tracks a change made after load', async () => {
  const emit = setSystemDark(false);
  const { seen, rerender } = await renderHook();
  assert.equal(seen.appearance, 'light');
  await rerender(() => emit(true));
  assert.equal(seen.appearance, 'dark', 'should follow the system switching to dark');
});

test('a corrupt stored value falls back to system rather than breaking', async () => {
  setSystemDark(true);
  window.localStorage.setItem('alethia.appearance', 'chartreuse');
  const { seen } = await renderHook();
  assert.equal(seen.preference, 'system');
  assert.equal(seen.appearance, 'dark');
});

test('the document colour-scheme is set so browser surfaces match', async () => {
  setSystemDark(true);
  await renderHook();
  assert.equal(document.documentElement.style.colorScheme, 'dark');
});
