// Show the worked example once and never again. The rule lives entirely in state and
// localStorage, so it is asserted here.
import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';

import { setupApp } from './app-harness.mjs';
import { EXAMPLES, DEFAULT_EXAMPLE } from '../src/ui/example.js';

// assert against whichever example opens, so changing the default is not a test edit
const opening = EXAMPLES[DEFAULT_EXAMPLE];

let act;
let mount;
let cleanup;

before(async () => {
  // Its own outDir: `node --test` runs files concurrently and two vite builds writing the
  // same directory would race.
  ({ act, mount, cleanup } = await setupApp({ outDir: '.itest-onboard' }));
});

after(() => cleanup?.());
beforeEach(() => window.localStorage.clear());

const listValues = (c) => [...c.querySelectorAll('textarea')].map((t) => t.value);

test('a first visit arrives with the worked example already in both lists', async () => {
  const { container, unmount } = await mount();
  const [messy, reference] = listValues(container);

  assert.equal(messy, opening.messy, 'the messy list should hold the example queries');
  assert.equal(
    reference,
    opening.reference,
    'the reference list should hold the canonical names',
  );
  assert.ok(
    container.textContent.includes('Example data'),
    'the example must announce itself as an example, or it reads as the user\'s own data',
  );
  await unmount();
});

test('the example does not come back on the next visit', async () => {
  const first = await mount();
  await first.unmount();
  assert.equal(window.localStorage.getItem('alethia.seen'), '1', 'the visit should be recorded');

  const second = await mount();
  assert.deepEqual(listValues(second.container), ['', ''], 'a returning visitor starts empty');
  assert.ok(!second.container.textContent.includes('Example data'));
  await second.unmount();
});

test('an empty page offers the example back', async () => {
  // Without this a returning visitor has no route back to the demonstration.
  window.localStorage.setItem('alethia.seen', '1');
  const { container, unmount } = await mount();
  assert.deepEqual(listValues(container), ['', '']);

  // a real button, not text styled to look like one; a ghost button in grey body text
  // reads as broken markup
  const load = [...container.querySelectorAll('button')]
    .find((b) => /load the example/i.test(b.textContent));
  assert.ok(load, 'the empty state must offer a button that loads the example');
  await unmount();
});

test('editing a list stops it being treated as the example', async () => {
  // "Is this still the example?" is derived from the list contents, not tracked in
  // a flag, so this is the assertion that keeps the notice honest: the moment the entries
  // are the user's own, the banner calling them a sample has to go.
  const { container, unmount } = await mount();
  assert.ok(container.textContent.includes('Example data'));

  const [messy] = container.querySelectorAll('textarea');
  await act(async () => {
    // React installs its own value setter on the element; going through the prototype's is
    // what registers the change instead of swallowing it as a no-op.
    const { set } = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value',
    );
    set.call(messy, 'Bombay\nCalcutta\nMadras');
    messy.dispatchEvent(new window.Event('input', { bubbles: true }));
  });

  assert.ok(
    !container.textContent.includes('Example data'),
    'the example notice must disappear once the entries are the user\'s own',
  );
  await unmount();
});

test('blocked storage shows the example rather than an empty page', async () => {
  // A managed browser can refuse localStorage outright, and this product's primary user is
  // often on one. Guessing "returning visitor" there would hand a first-time user two
  // empty boxes and no idea what the tool does, which is the worse of the two mistakes.
  const own = Object.getOwnPropertyDescriptor(window, 'localStorage');
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    get() { throw new Error('storage blocked'); },
  });
  try {
    const { container, unmount } = await mount();
    assert.equal(listValues(container)[0], opening.messy);
    await unmount();
  } finally {
    if (own) Object.defineProperty(window, 'localStorage', own);
    else delete window.localStorage;
  }
});

test('an insecure origin warns that the model will not be cached', async () => {
  // transformers.js keeps models in Cache Storage, which is absent outside a secure
  // context, so http://<lan-ip> silently re-downloads on every reload
  const own = Object.getOwnPropertyDescriptor(window, 'isSecureContext');
  Object.defineProperty(window, 'isSecureContext', { configurable: true, value: false });
  try {
    const { container, unmount } = await mount();
    assert.ok(
      container.textContent.includes('not on a secure origin'),
      'a page that cannot cache the model has to say so',
    );
    await unmount();
  } finally {
    // jsdom leaves isSecureContext undefined and non-own, so there is nothing to put
    // back; leaving the stub in place leaks a false into the next test
    if (own) Object.defineProperty(window, 'isSecureContext', own);
    else delete window.isSecureContext;
  }
});

test('a secure origin says nothing about caching', async () => {
  const { container, unmount } = await mount();
  assert.ok(!container.textContent.includes('not on a secure origin'));
  await unmount();
});
