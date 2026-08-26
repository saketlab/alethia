import { Badge, Code, Flex, Table, Text } from '@radix-ui/themes';
import { CheckIcon } from '@radix-ui/react-icons';

/**
 * The "why" behind the recommendation.
 *
 * Each row states what the metric asks and which direction is better, so a reader can
 * disagree with the recommendation for a reason.
 */
export const DESCRIPTIONS = {
  centered_nn_similarity: {
    label: 'Reference crowding',
    better: 'lower',
    text: 'How close distinct reference entries sit to each other once the shared '
      + 'direction is removed. Crowded references are easy to confuse.',
  },
  mean_margin_z: {
    label: 'Decisiveness',
    better: 'higher',
    text: 'How far the best match stands above a typical one, in standard deviations. '
      + 'Scale-free, so it compares fairly across models.',
  },
  low_margin_z_rate: {
    label: 'Ambiguous matches',
    better: 'lower',
    text: 'Fraction of entries where the top match barely beat the runner-up.',
  },
  mutual_nn_rate: {
    label: 'Agreement',
    better: 'higher',
    text: 'How often the chosen reference also picks your entry back. Agreement in '
      + 'both directions is a stronger signal than one-way similarity.',
  },
  hubness_skew: {
    label: 'Hub distortion',
    better: 'lower',
    text: 'Whether a few references absorb disproportionately many matches, which is a '
      + 'known failure mode of high-dimensional embeddings.',
  },
};

// listed in descending composite weight, so the top rows decided the outcome
export const SHOWN = [
  'centered_nn_similarity', 'mean_margin_z', 'mutual_nn_rate',
  'hubness_skew', 'low_margin_z_rate',
];

export default function MetricTable({ metrics, best }) {
  const models = Object.keys(metrics);

  return (
    <Flex direction="column" gap="3">
      <Text size="2" color="gray" style={{ maxWidth: '64ch' }}>
        Measured on your data, with no correct answers needed. The recommendation
        combines them, weighted so related metrics cannot count twice.
      </Text>
      <Table.Root variant="surface" size="2">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeaderCell>What it measures</Table.ColumnHeaderCell>
            {models.map((m) => (
              <Table.ColumnHeaderCell key={m} align="right">
                <Flex align="center" justify="end" gap="2">
                  {m}
                  {m === best && <Badge size="1" color="indigo" radius="full">best</Badge>}
                </Flex>
              </Table.ColumnHeaderCell>
            ))}
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {SHOWN.filter((key) => models.some((m) => Number.isFinite(metrics[m][key]))).map((key) => {
            const meta = DESCRIPTIONS[key];
            const values = models.map((m) => metrics[m][key]);
            // negating turns lowest-absolute into highest-score, so one comparison
            // serves every row; a NaN cell loses either way
            const rank = (v) => (meta.better === 'higher' ? v : -Math.abs(v));
            const top = Math.max(...values.filter(Number.isFinite).map(rank));

            return (
              <Table.Row key={key}>
                <Table.RowHeaderCell>
                  {/* These explanations used to live in a hover tooltip on a plain span:
                      unreachable by keyboard, invisible on touch, and hidden from exactly
                      the reader who most needs them. They are the content of this table,
                      not an aside, so they are on the page. */}
                  <Flex direction="column" gap="1" style={{ maxWidth: '48ch' }}>
                    <Flex align="center" gap="2" wrap="wrap">
                      <Text weight="medium">{meta.label}</Text>
                      <Badge size="1" variant="soft" color="gray" radius="full">
                        {meta.better} is better
                      </Badge>
                    </Flex>
                    <Text size="1" color="gray">{meta.text}</Text>
                  </Flex>
                </Table.RowHeaderCell>
                {models.map((m, i) => {
                  const v = values[i];
                  const isWinner = rank(v) === top;
                  return (
                    <Table.Cell key={m} align="right">
                      {/* The winning cell was distinguished by colour alone. The check
                          mark and the hidden word carry the same fact without it. */}
                      <Flex align="center" justify="end" gap="1">
                        {isWinner && <CheckIcon aria-hidden color="var(--accent-11)" />}
                        <Code
                          variant={isWinner ? 'soft' : 'ghost'}
                          color={isWinner ? 'indigo' : 'gray'}
                          className="tabular"
                        >
                          {Number.isFinite(v) ? v.toFixed(3) : '-'}
                        </Code>
                        {isWinner && <span className="visually-hidden">best in this row</span>}
                      </Flex>
                    </Table.Cell>
                  );
                })}
              </Table.Row>
            );
          })}
        </Table.Body>
      </Table.Root>
      <Text size="1" color="gray">
        A check marks the model that wins that row. Metrics are weighted by family, so
        a model can lead on count and still lose overall.
      </Text>
    </Flex>
  );
}
