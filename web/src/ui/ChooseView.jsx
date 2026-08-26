import { memo, useMemo, useState } from 'react';
import {
  Badge, Button, Callout, Card, Checkbox, Code, Flex, Grid, Heading, Table, Text,
} from '@radix-ui/themes';
import {
  ArrowRightIcon, CheckCircledIcon, InfoCircledIcon, MagnifyingGlassIcon,
} from '@radix-ui/react-icons';

import MetricTable, { DESCRIPTIONS, SHOWN } from './MetricTable.jsx';
import { MetricChart, ScoreChart } from './Charts.jsx';
import { MODELS } from './models.js';

/**
 * Compare models on the user's own data and recommend one.
 *
 * The recommendation ends in a button that carries the winning model to the Match view
 * by URL.
 */
export default function ChooseView({ queries, references, runJob, busy, navigate }) {
  const [selected, setSelected] = useState(
    () => MODELS.filter((m) => m.default).map((m) => m.id),
  );
  const [result, setResult] = useState(null);

  const chosen = useMemo(
    () => MODELS.filter((m) => selected.includes(m.id)),
    [selected],
  );
  const ready = queries.length > 0 && references.length > 1 && chosen.length >= 2 && !busy;

  const toggle = (id) => setSelected(
    selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id],
  );

  const run = async () => {
    setResult(null);
    const payload = await runJob({
      type: 'assess',
      payload: { queries, references, models: chosen },
    }).catch(() => null);
    if (payload) setResult(payload);
  };

  return (
    <Flex direction="column" gap="4">
      <Card size="3" variant="surface">
        <Flex direction="column" gap="3">
          <Flex align="baseline" justify="between" gap="3" wrap="wrap">
            <Heading as="h2" size="4" weight="medium">Models to compare</Heading>
            <Text size="1" color="gray" className="tabular">
              {selected.length} selected | downloaded once, then cached
            </Text>
          </Flex>
          <Text size="2" color="gray" style={{ maxWidth: '64ch' }}>
            Pick at least two. Each one is downloaded and run over your references,
            so more models take longer.
          </Text>

          <Grid columns={{ initial: '1', sm: '2' }} gap="2" pt="1">
            {MODELS.map((model) => (
              <Text as="label" size="2" key={model.id}>
                <Flex gap="2" align="start">
                  <Checkbox
                    checked={selected.includes(model.id)}
                    onCheckedChange={() => toggle(model.id)}
                  />
                  <Flex direction="column" gap="1" style={{ minWidth: 0 }}>
                    <Flex align="center" gap="2">
                      <Text weight="medium">{model.label}</Text>
                      <Badge size="1" variant="soft" color="gray" radius="full">
                        {model.size}
                      </Badge>
                    </Flex>
                    <Text size="1" color="gray">{model.note}</Text>
                  </Flex>
                </Flex>
              </Text>
            ))}
          </Grid>

          <Flex justify="between" align="center" gap="3" pt="2" wrap="wrap">
            <Text size="1" color="gray" className="tabular">
              {chosen.reduce((mb, m) => mb + parseInt(m.size, 10), 0)} MB to download
              on first run
            </Text>
            <Button size="3" disabled={!ready} onClick={run}>
              <MagnifyingGlassIcon aria-hidden /> Compare {chosen.length} models
            </Button>
          </Flex>

          {!ready && !busy && (
            <Callout.Root color="gray" variant="surface" size="1">
              <Callout.Icon><InfoCircledIcon /></Callout.Icon>
              <Callout.Text>{hint(queries, references, chosen)}</Callout.Text>
            </Callout.Root>
          )}
        </Flex>
      </Card>

      {result && <Recommendation result={result} references={references} navigate={navigate} />}
    </Flex>
  );
}

function hint(queries, references, chosen) {
  if (!queries.length) return 'Add the messy entries you want cleaned up.';
  if (references.length < 2) return 'Add at least two correct entries to compare against.';
  if (chosen.length < 2) return 'Select at least two models so they can be ranked.';
  return '';
}

// result is fixed once the comparison finishes, but the shell re-renders under it
const Recommendation = memo(function Recommendation({ result, references, navigate }) {
  const { ranking, best, bestId, metrics } = result;

  return (
    <Flex direction="column" gap="4">
      <Card size="4" variant="surface">
        <Flex direction="column" gap="3">
          <Flex align="center" gap="2">
            <CheckCircledIcon color="var(--accent-9)" aria-hidden />
            <Text size="2" color="gray" weight="medium">Best model for your data</Text>
          </Flex>
          <Heading as="h2" size="8" weight="bold" trim="both">{best}</Heading>
          <Text size="2" color="gray" style={{ maxWidth: '64ch' }}>
            Chosen by comparing the geometry each model produces on your{' '}
            {references.length.toLocaleString()} reference entries, with no correct
            answers needed. The scores rank these models against each other.
          </Text>
          <Flex pt="2">
            {/* The step the old single-button flow left implicit. */}
            <Button size="3" onClick={() => navigate('match', { model: bestId })}>
              Use {best} to match <ArrowRightIcon aria-hidden />
            </Button>
          </Flex>
        </Flex>
      </Card>

      <Card size="3" variant="surface">
        <Flex direction="column" gap="3">
          <Heading as="h2" size="3" weight="medium">Ranking</Heading>
          <ScoreChart ranking={ranking} best={best} />
          <Table.Root variant="surface" size="2">
            <Table.Header>
              <Table.Row>
                <Table.ColumnHeaderCell>Model</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell align="right">Composite</Table.ColumnHeaderCell>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {ranking.map((row) => (
                <Table.Row key={row.model}>
                  <Table.RowHeaderCell>
                    <Flex align="center" gap="2">
                      {row.model}
                      {row.model === best && (
                        <Badge color="indigo" variant="soft" radius="full">
                          recommended
                        </Badge>
                      )}
                    </Flex>
                  </Table.RowHeaderCell>
                  <Table.Cell align="right">
                    <Code variant="ghost" className="tabular">
                      {Number.isFinite(row.score) ? row.score.toFixed(3) : '-'}
                    </Code>
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Root>
          {/* Was a hover tooltip on the column header, which no keyboard could reach.
              It is the sentence that stops someone reading 0.412 as an accuracy. */}
          <Text size="1" color="gray">
            Higher is better. The composite is relative to the models you compared,
            not an accuracy.
          </Text>
        </Flex>
      </Card>

      <Card size="3" variant="surface">
        <Flex direction="column" gap="3">
          <Heading as="h2" size="3" weight="medium">Why</Heading>
          <MetricChart
            metrics={metrics}
            best={best}
            shown={SHOWN}
            descriptions={DESCRIPTIONS}
          />
          <MetricTable metrics={metrics} best={best} />
        </Flex>
      </Card>
    </Flex>
  );
});
