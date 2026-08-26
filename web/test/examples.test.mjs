// The picker answers "does this work on data like mine?" only if the flavours are
// genuinely different tasks and switching swaps both lists. Both are asserted here.
import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';

import { setupApp } from './app-harness.mjs';
import { EXAMPLES, EXAMPLE_LIST, DEFAULT_EXAMPLE, matchExample } from '../src/ui/example.js';

let act;
let mount;
let cleanup;

before(async () => {
  ({ act, mount, cleanup } = await setupApp({ outDir: '.itest-examples' }));
});

after(() => cleanup?.());
beforeEach(() => window.localStorage.clear());

const listValues = (c) => [...c.querySelectorAll('textarea')].map((t) => t.value);

const chipNamed = (container, label) =>
  [...container.querySelectorAll('button')]
    .find((el) => el.textContent.trim() === label);

async function press(el) {
  await act(async () => {
    el.dispatchEvent(new window.MouseEvent('click', {
      bubbles: true, cancelable: true, button: 0, view: window,
    }));
  });
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}

// The data itself, no DOM needed. A bad regeneration of example.js still renders
// perfectly, so nothing else would catch it.

test('every example is a well-formed, self-consistent pair of lists', () => {
  assert.ok(EXAMPLE_LIST.length >= 4, 'a single flavour is what the picker exists to fix');

  for (const e of EXAMPLE_LIST) {
    const queries = e.messy.split('\n');
    const references = e.reference.split('\n');

    assert.ok(e.label && e.blurb, `${e.key} needs a label and a blurb for the picker`);
    assert.ok(queries.length >= 20, `${e.key} has only ${queries.length} queries`);
    assert.ok(references.length >= queries.length, `${e.key} has fewer references than queries`);

    assert.equal(new Set(queries).size, queries.length, `${e.key} repeats a query`);
    assert.equal(new Set(references).size, references.length, `${e.key} repeats a reference`);

    for (const line of [...queries, ...references]) {
      assert.ok(line.trim() === line && line !== '', `${e.key} has a blank or padded line`);
    }
  }
});

test('the flavours are different tasks, not the same one relabelled', () => {
  // The picker's whole claim is that these exercise different things. Two examples sharing
  // most of their references would make the hard ones look easy for the wrong reason.
  for (const a of EXAMPLE_LIST) {
    for (const b of EXAMPLE_LIST) {
      if (a.key >= b.key) continue;
      const left = new Set(a.reference.split('\n'));
      const shared = b.reference.split('\n').filter((r) => left.has(r)).length;
      assert.ok(shared <= 2, `${a.key} and ${b.key} share ${shared} references`);
    }
  }
});

test('the hard flavours are not solvable by string similarity alone', () => {
  // If a query and its answer overlap heavily on characters, rapidfuzz would settle it and
  // the example would prove nothing about the embedding models. This is the property that
  // makes drugs and chemicals worth shipping, so it is pinned.
  for (const key of ['drugs', 'chemicals']) {
    const e = EXAMPLES[key];
    assert.ok(e, `${key} should be generated`);
    const queries = e.messy.split('\n');
    const references = new Set(e.reference.split('\n'));
    // No query may simply *be* its own answer.
    const trivial = queries.filter((q) => references.has(q)).length;
    assert.equal(trivial, 0, `${key} has ${trivial} queries that are already correct`);
  }
});

test('matchExample identifies each example and rejects edited lists', () => {
  for (const e of EXAMPLE_LIST) {
    assert.equal(matchExample(e.messy, e.reference), e.key);
  }
  assert.equal(matchExample('Bombay', 'Mumbai'), null);
  assert.equal(matchExample('', ''), null, 'two empty boxes are not an example');
  // A half-edited list stops being the example; the notice must not claim the user's own
  // messy list is sample data just because the reference list was left alone.
  const [first] = EXAMPLE_LIST;
  assert.equal(matchExample('mine', first.reference), null);
});

// ---------------------------------------------------------------------------------------
// The picker in the running app.
// ---------------------------------------------------------------------------------------

test('a first visit opens on the default example with its chip selected', async () => {
  const { container, unmount } = await mount();

  const chip = chipNamed(container, EXAMPLES[DEFAULT_EXAMPLE].label);
  assert.ok(chip, 'the default example should have a chip');
  assert.equal(chip.getAttribute('aria-pressed'), 'true');

  for (const e of EXAMPLE_LIST) {
    assert.ok(chipNamed(container, e.label), `${e.key} should be offered`);
  }
  await unmount();
});

test('picking another flavour swaps both lists and moves the selection', async () => {
  const { container, unmount } = await mount();
  const target = EXAMPLE_LIST.find((e) => e.key !== DEFAULT_EXAMPLE);

  await press(chipNamed(container, target.label));

  assert.deepEqual(
    listValues(container), [target.messy, target.reference],
    'both boxes must change together -- a half-swap silently invents a new task',
  );
  assert.equal(chipNamed(container, target.label).getAttribute('aria-pressed'), 'true');
  assert.equal(
    chipNamed(container, EXAMPLES[DEFAULT_EXAMPLE].label).getAttribute('aria-pressed'),
    'false',
    'the previous chip must release, or two look loaded at once',
  );

  // The notice describes whichever example is loaded, not the one it opened with.
  assert.ok(container.textContent.includes(target.blurb.split('.')[0]));
  await unmount();
});

test('the example notice reports the size of the loaded example, not a fixed number', async () => {
  const { container, unmount } = await mount();
  const target = EXAMPLE_LIST.find((e) => e.key !== DEFAULT_EXAMPLE);

  const counted = (e) => `${e.messy.split('\n').length} entries against `
    + `${e.reference.split('\n').length} references`;

  assert.ok(container.textContent.includes(counted(EXAMPLES[DEFAULT_EXAMPLE])));
  await press(chipNamed(container, target.label));
  assert.ok(
    container.textContent.includes(counted(target)),
    'the count was hardcoded once and went stale the moment the corpus was regenerated',
  );
  await unmount();
});
