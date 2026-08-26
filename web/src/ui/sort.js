/**
 * Ordering for the results table.
 *
 * Plain JS, so the edge cases (unmatched rows, a genuine zero score, locale-aware
 * text) can be tested without standing up a DOM.
 */

export const SORT_LABELS = { given: 'your entry', prediction: 'match', score: 'score' };

export function sortValue(m, key) {
  if (key === 'score') return m.alethia_score ?? null;
  // || not ??; '' would sort to the top and read as a row that matched blank
  if (key === 'prediction') return m.alethia_prediction || null;
  return m.given_entity ?? '';
}

/**
 * First activation sorts a column, second reverses it, third returns to review order.
 *
 * The third state exists because the review order, corrected rows first, is not any
 * column's sort; a two-state toggle could never get back to it.
 */
export function cycleSort(prev, key) {
  if (prev?.key !== key) return { key, dir: 'asc' };
  return prev.dir === 'asc' ? { key, dir: 'desc' } : null;
}

/**
 * Order the whole result set, not the visible page.
 *
 * Sorting only the preview rows would rank the visible sample, not the result set.
 *
 * Returns the input array unchanged when there is no sort, so the caller's memo keeps
 * its identity.
 */
export function sortRows(ordered, sort) {
  if (!sort) return ordered;
  const { key, dir } = sort;
  const sign = dir === 'asc' ? 1 : -1;
  return [...ordered].sort((a, b) => {
    const av = sortValue(a, key);
    const bv = sortValue(b, key);
    // unmatched rows stay at the bottom in both directions, not just ascending
    if (av === null || bv === null) {
      if (av === bv) return 0;
      return av === null ? 1 : -1;
    }
    return sign * (typeof av === 'number' ? av - bv : av.localeCompare(bv));
  });
}
