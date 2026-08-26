import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { renderToString } from 'react-dom/server';
import React from 'react';
import { Theme } from '@radix-ui/themes';
import { build } from 'vite';

// The router touches window during render; supply just enough of it for SSR.
global.window = {
  location: { hash: '', origin: 'http://localhost', pathname: '/' },
  addEventListener() {}, removeEventListener() {},
  history: { replaceState() {} },
};
// navigator is read-only in Node 24 and is not touched during render, so it is left alone.

// Compile the JSX through vite (same transform the browser gets), then render it to a
// string. A clean build only proves it parses; this proves the component tree mounts.
mkdirSync('./.ssr-check', { recursive: true });
writeFileSync('./.ssr-check/entry.jsx', `
  export { default as App } from '${process.cwd()}/src/ui/App.jsx';
`);
await build({
  root: process.cwd(),
  logLevel: 'error',
  build: {
    ssr: './.ssr-check/entry.jsx',
    outDir: './.ssr-check/out',
    rollupOptions: { external: ['react', 'react-dom', '@radix-ui/themes', '@radix-ui/react-icons'] },
  },
});
const { App } = await import('./.ssr-check/out/entry.js');
let missing = 0;
function render(hash, label, checks) {
  // Radix Tabs only renders the active tab, so each route is rendered separately.
  global.window.location.hash = hash;
  const html = renderToString(React.createElement(
    Theme, null,
    React.createElement(App, { appearance: { preference: 'system', choose() {} } }),
  ));
  console.log(`  [${label}] ${html.length} chars`);
  for (const c of checks) {
    const ok = html.includes(c);
    if (!ok) missing++;
    console.log(`    ${ok ? 'ok  ' : 'MISS'}  ${c}`);
  }

  // Structure, not strings. Radix's <Heading> defaults to as="h1", so a call site that
  // forgets `as` silently emits a second top-level heading and flattens the outline that
  // screen-reader users navigate by, a convention no reviewer reliably catches. Landmarks
  // fail the same silent way. Asserting on the rendered HTML turns both into build gates.
  for (const [assertion, ok] of [
    [`exactly one <h1> (found ${(html.match(/<h1[\s>]/g) || []).length})`,
      (html.match(/<h1[\s>]/g) || []).length === 1],
    ['<main> landmark', /<main[\s>]/.test(html)],
    ['<header> landmark', /<header[\s>]/.test(html)],
    ['<footer> landmark', /<footer[\s>]/.test(html)],
    ['a polite live region', /aria-live="polite"/.test(html)],
    ['an alert region', /role="alert"/.test(html)],
  ]) {
    if (!ok) missing++;
    console.log(`    ${ok ? 'ok  ' : 'MISS'}  ${assertion}`);
  }
}

render('#/match', 'match', [
  'alethia', 'Messy entries', 'Correct entries', 'never leave this device',
  'alethia on GitHub', 'github.com/saketlab/alethia',
  'Run ', 'Minimum score', 'Choose a model',
  'System', 'Light', 'Dark',
]);
render('#/choose', 'choose', [
  'Models to compare', 'Compare ', 'MiniLM-L6', 'Pick at least two',
]);

console.log(`  ${missing} missing`);
process.exit(missing ? 1 : 0);
