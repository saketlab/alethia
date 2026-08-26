// Shared bootstrap for the suites that mount the real App in jsdom. The NODE_ENV
// ordering constraint below has to exist once; two copies drift on the next upgrade.
import assert from 'node:assert/strict';

/**
 * Build the app through vite and stand up a DOM for it.
 *
 * `outDir` must differ per suite: `node --test` runs files concurrently, and two builds
 * writing the same directory race.
 */
export async function setupApp({ outDir }) {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const jsdom = (await import('global-jsdom')).default;
  const cleanup = jsdom('<div id="root"></div>', {
    url: 'http://localhost/', pretendToBeVisual: true,
  });

  // jsdom implements none of these; Radix's Select and ScrollArea use them for
  // positioning. Stubs are enough because layout is not what these tests assert on.
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub;
  window.ResizeObserver = ResizeObserverStub;
  window.HTMLElement.prototype.scrollIntoView = () => {};
  if (!window.matchMedia) {
    window.matchMedia = () => ({
      matches: false, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {},
    });
  }

  // vite's build() sets NODE_ENV=production, and React's conditional exports then resolve
  // to the production build, which does not export act(). Restore the environment before
  // importing React, or every test fails on a missing act.
  const priorNodeEnv = process.env.NODE_ENV;
  const { build } = await import('vite');
  await build({
    root: process.cwd(),
    logLevel: 'error',
    build: {
      ssr: 'src/ui/App.jsx',
      outDir,
      rollupOptions: {
        external: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime',
          '@radix-ui/themes', '@radix-ui/react-icons'],
      },
    },
  });

  process.env.NODE_ENV = priorNodeEnv ?? 'development';
  const React = await import('react');
  const { act } = React;
  assert.equal(typeof act, 'function', 'React.act must be available (development build)');
  const { createRoot } = await import('react-dom/client');
  const App = (await import(`${process.cwd()}/${outDir}/App.js`)).default;
  const { Theme } = await import('@radix-ui/themes');

  /** Mount the app at a route and return its container plus an unmount. */
  async function mount(hash = '') {
    window.location.hash = hash;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    // Radix components read theme context, so the app mounts inside Theme;
    // exactly as main.jsx does it.
    await act(async () => {
      root.render(React.createElement(
        Theme, null,
        React.createElement(App, { appearance: { preference: 'system', choose() {} } }),
      ));
    });
    return { container, unmount: () => act(() => root.unmount()) };
  }

  return { React, act, mount, cleanup };
}

/**
 * Activate a tab the way a person does.
 *
 * element.click() dispatches only a click event. Radix activates a tab on mousedown and
 * on focus, so a bare click does nothing, the component is not broken, the synthetic
 * event is just not what it listens for.
 */
export async function activate(act, el) {
  await act(async () => {
    el.focus();
    for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
      el.dispatchEvent(new window.MouseEvent(type, {
        bubbles: true, cancelable: true, button: 0, view: window,
      }));
    }
  });
  // Let the hashchange listener and any resulting effect settle.
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}
