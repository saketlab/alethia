/**
 * Bar lengths for a set of model values, always oriented so longer is better.
 *
 * Metrics point in different directions and sit on different scales, so plotting them
 * as measured would put the winning model at the short end of half the charts. Each set
 * is rescaled across the compared models and flipped when lower is better. A floor keeps
 * the weakest model visible.
 */
export function barFractions(values, { better = 'higher', floor = 0.08 } = {}) {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return values.map(() => 0);
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const span = hi - lo || 1;
  return values.map((v) => {
    if (!Number.isFinite(v)) return 0;
    const unit = (v - lo) / span;
    const oriented = better === 'lower' ? 1 - unit : unit;
    return floor + (1 - floor) * oriented;
  });
}
