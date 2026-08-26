// Tab clicks are asserted against a real DOM, driven through react-dom/client and
// React.act so there is one React instance. The views compile through vite first.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';

import { activate, setupApp } from './app-harness.mjs';

let act;
let mount;
let cleanup;

before(async () => {
  ({ act, mount, cleanup } = await setupApp({ outDir: '.itest' }));
});

after(() => cleanup?.());

// Radix renders a trigger's label twice, once visible, once as a hidden spacer that
// reserves the width of the bold selected state, so textContent reads "MatchMatch".
const tabNamed = (container, label) =>
  [...container.querySelectorAll('[role="tab"]')]
    .find((el) => el.textContent.trim().toLowerCase().startsWith(label));

test('clicking "Choose a model" switches to that view and stays there', async () => {
  const { container, unmount } = await mount('');
  const tab = tabNamed(container, 'choose a model');
  assert.ok(tab, 'the tab should exist');

  await activate(act, tab);

  // The symptom to catch: an effect rewriting the hash back to #/match would bounce the
  // user straight out of the view they just opened.
  assert.equal(window.location.hash, '#/choose', 'hash should stay on the choose route');
  assert.ok(
    container.textContent.includes('Models to compare'),
    'the choose view content should be visible',
  );
  await unmount();
});

test('clicking back to Match returns to the match view', async () => {
  const { container, unmount } = await mount('#/choose');
  await activate(act, tabNamed(container, 'match'));

  assert.ok(window.location.hash.startsWith('#/match'), `got ${window.location.hash}`);
  assert.ok(
    container.textContent.includes('Minimum score'),
    'the match view should be visible',
  );
  await unmount();
});

test('opening #/choose directly does not redirect away', async () => {
  const { container, unmount } = await mount('#/choose');
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  assert.equal(window.location.hash, '#/choose');
  assert.ok(container.textContent.includes('Models to compare'));
  await unmount();
});
