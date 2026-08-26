import { useCallback, useEffect, useMemo, useState } from 'react';

/**
 * Hash routing.
 *
 * The History API needs the server to rewrite every path back to index.html; a hash
 * route survives a refresh on any static host with no server config.
 *
 * Routes:
 *   #/match?model=<id>   run one model over your lists
 *   #/choose             compare models and get a recommendation
 */

const DEFAULT_ROUTE = { view: 'match', params: {} };

export function parseHash(hash = window.location.hash) {
  const raw = hash.replace(/^#\/?/, '');
  if (!raw) return DEFAULT_ROUTE;
  const [path, query = ''] = raw.split('?');
  const view = ['match', 'choose'].includes(path) ? path : DEFAULT_ROUTE.view;
  return { view, params: Object.fromEntries(new URLSearchParams(query)) };
}

export function buildHash(view, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v != null && v !== ''),
  ).toString();
  return `#/${view}${query ? `?${query}` : ''}`;
}

/** Current route, kept in sync with back/forward and manual edits to the address bar. */
export function useRoute() {
  const [hash, setHash] = useState(() => window.location.hash);

  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  // keyed on the hash so the object keeps identity; a fresh one breaks dependent effects
  const route = useMemo(() => parseHash(hash), [hash]);

  // stable across renders; a fresh function re-runs every navigating effect
  const navigate = useCallback((view, params = {}, { replace = false } = {}) => {
    const next = buildHash(view, params);
    if (next === window.location.hash) return;
    if (replace) {
      window.history.replaceState(null, '', next);
      setHash(next);
    } else {
      window.location.hash = next;
    }
  }, []);

  return [route, navigate];
}

/** Absolute URL for the current route, for the copy-link affordance. */
export function shareableUrl(view, params = {}) {
  const { origin, pathname } = window.location;
  return `${origin}${pathname}${buildHash(view, params)}`;
}
