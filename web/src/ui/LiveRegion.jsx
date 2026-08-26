/**
 * A screen-reader announcement channel.
 *
 * Always render it, even with nothing to say: a region inserted in the same commit as
 * its first message is generally not announced, and the failure is silent.
 *
 * Hidden channel only; a visible region carries its own role.
 */
export function LiveRegion({ children }) {
  return (
    <span className="visually-hidden" role="status" aria-live="polite">
      {children}
    </span>
  );
}
