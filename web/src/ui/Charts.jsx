import { Flex, Text } from '@radix-ui/themes';
import { barFractions } from './chart-scale.js';

/**
 * Bars for the composite ranking and for each metric behind it.
 *
 * Inline SVG, since every shape here is a rectangle and the bundle already carries a
 * wasm runtime. Colours come from Radix scale tokens, so both charts follow the theme.
 */

const BAR = 22;
const GAP = 8;
const LABEL_W = 116;

function Bars({ rows, title, width = 420 }) {
  const height = rows.length * (BAR + GAP);
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      role="img"
      aria-label={title}
      style={{ overflow: 'visible' }}
    >
      {rows.map((r, i) => {
        const y = i * (BAR + GAP);
        const track = width - LABEL_W - 46;
        return (
          <g key={r.label}>
            <text
              x={LABEL_W - 8}
              y={y + BAR / 2}
              textAnchor="end"
              dominantBaseline="central"
              fontSize="12"
              fill={r.best ? 'var(--accent-11)' : 'var(--gray-11)'}
              fontWeight={r.best ? 600 : 400}
            >
              {r.label}
            </text>
            <rect
              x={LABEL_W} y={y} width={track} height={BAR} rx="4"
              fill="var(--gray-4)"
            />
            <rect
              x={LABEL_W} y={y} width={Math.max(2, track * r.fraction)} height={BAR} rx="4"
              fill={r.best ? 'var(--accent-9)' : 'var(--gray-8)'}
            />
            <text
              x={LABEL_W + track + 8}
              y={y + BAR / 2}
              dominantBaseline="central"
              fontSize="12"
              fill="var(--gray-11)"
              className="tabular"
            >
              {r.readout}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Composite score per model, scaled across the models compared. */
export function ScoreChart({ ranking, best }) {
  const fractions = barFractions(ranking.map((r) => r.score));
  if (fractions.every((f) => f === 0)) return null;

  const rows = ranking.map((r, i) => ({
    label: r.model,
    best: r.model === best,
    fraction: fractions[i],
    readout: Number.isFinite(r.score) ? r.score.toFixed(3) : '-',
  }));

  return (
    <Flex direction="column" gap="2">
      <Bars rows={rows} title="Composite score per model, best first" />
      <Text size="1" color="gray">
        Bar length is relative to the models you compared, not an accuracy.
      </Text>
    </Flex>
  );
}

/**
 * One small chart per metric, every bar oriented so longer is better.
 *
 * Raw metrics point in different directions and sit on different scales, so plotting
 * them as measured would put the winning model at the short end of half the charts.
 * Each is rescaled across the compared models and flipped when lower is better.
 */
export function MetricChart({ metrics, best, shown, descriptions }) {
  const models = Object.keys(metrics);
  if (models.length < 2) return null;

  return (
    <Flex direction="column" gap="4">
      {shown.map((key) => {
        const d = descriptions[key];
        const vals = models.map((m) => metrics[m]?.[key]);
        const fractions = barFractions(vals, { better: d.better });
        if (fractions.every((f) => f === 0)) return null;

        const rows = models.map((m, i) => ({
          label: m,
          best: m === best,
          fraction: fractions[i],
          readout: Number.isFinite(vals[i]) ? vals[i].toFixed(3) : '-',
        }));

        return (
          <Flex key={key} direction="column" gap="1">
            <Text size="2" weight="medium">
              {d.label}{' '}
              <Text size="1" color="gray" weight="regular">
                ({d.better} is better)
              </Text>
            </Text>
            <Bars rows={rows} title={`${d.label}, ${d.better} is better`} width={420} />
          </Flex>
        );
      })}
    </Flex>
  );
}
