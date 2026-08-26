import { useCallback, useEffect, useState } from 'react';

/**
 * Light and dark, following the operating system by default.
 *
 * Three states. A plain toggle forces a choice the user has usually
 * already made system-wide, and then ignores it when they switch at sunset; "system"
 * keeps following. An explicit light or dark choice is remembered, because someone who
 * overrides the system means it.
 */
const STORAGE_KEY = 'alethia.appearance';
const PREFERENCES = ['system', 'light', 'dark'];

function readStored() {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return PREFERENCES.includes(value) ? value : 'system';
  } catch {
    // private browsing and blocked storage both throw; following the system is fine
    return 'system';
  }
}

function systemAppearance() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useAppearance() {
  const [preference, setPreference] = useState(readStored);
  const [system, setSystem] = useState(systemAppearance);

  // follow later system changes too, not just the value at load
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!query) return undefined;
    const onChange = (event) => setSystem(event.matches ? 'dark' : 'light');
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  const choose = useCallback((next) => {
    setPreference(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage being unavailable only costs persistence across reloads.
    }
  }, []);

  const appearance = preference === 'system' ? system : preference;

  // without this the page paints white for a moment in dark mode
  useEffect(() => {
    document.documentElement.style.colorScheme = appearance;
  }, [appearance]);

  return { preference, appearance, choose };
}
