import { memo, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge, Box, Button, Callout, Card, Flex, Heading, IconButton, Select, Table, Text,
  TextField, Tooltip,
} from '@radix-ui/themes';
import {
  ArrowDownIcon, ArrowUpIcon, CaretSortIcon, CheckIcon, DownloadIcon, InfoCircledIcon,
  Link2Icon, PlayIcon,
} from '@radix-ui/react-icons';

import { DEFAULT_THRESHOLD, MODELS } from './models.js';
import { entryLabel } from './ListInput.jsx';
import { downloadCsv } from './io.js';
import { SORT_LABELS, cycleSort, sortRows } from './sort.js';
import { shareableUrl } from './router.js';

/**
 * Run one model, get corrected entries.
 *
 * The model lives in the URL, so a run is reproducible by link. The data does not; it
 * never leaves the machine.
 */
export default function MatchView({
  queries, references, runJob, busy, navigate, modelId, route,
}) {
  const [result, setResult] = useState(null);
  // matches DEFAULT_THRESHOLD in the Python and R packages; blank means best guess
  const [threshold, setThreshold] = useState(String(DEFAULT_THRESHOLD));
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef(null);

  const selected = useMemo(
    () => MODELS.find((m) => m.id === modelId) ?? MODELS.find((m) => m.default),
    [modelId],
  );

  // the view guard is load-bearing: Radix keeps the inactive panel mounted, so an
  // unguarded effect rewrites the hash and bounces the user back. Deps stay primitive;
  // parseHash rebuilds route.params every render.
  useEffect(() => {
    if (route.view === 'match' && !modelId && selected) {
      navigate('match', { model: selected.id }, { replace: true });
    }
  }, [route.view, modelId, selected?.id, navigate]);

  useEffect(() => () => clearTimeout(copyTimer.current), []);

  const ready = queries.length > 0 && references.length > 0 && !busy;

  const run = async () => {
    setResult(null);
    // a NaN reaching the worker compares false against every score, silencing the run
    const typed = threshold === '' ? null : Number(threshold);
    const used = Number.isFinite(typed) ? typed : null;
    // the worker echoes the cutoff it ran with, so the table cannot mislabel it
    const payload = await runJob({
      type: 'match',
      payload: { queries, references, model: selected, threshold: used },
    }).catch(() => null);
    if (payload) setResult(payload);
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareableUrl('match', { model: selected.id }));
    } catch {
      // refused outside a secure context and under some managed browser policies
      return;
    }
    setCopied(true);
    clearTimeout(copyTimer.current);
    copyTimer.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Flex direction="column" gap="4">
      <Card size="3" variant="surface">
        <Flex direction="column" gap="4">
          <Flex gap="4" align="start" wrap="wrap">
            {/* flex-basis sets the preferred width; minWidth 0 lets it shrink below that.
                A hard min-width here overflowed the card on a 320px phone, where the
                container's padding plus the card's 24px leaves only ~240px. */}
            <Box style={{ flex: '1 1 260px', minWidth: 0 }}>
              <Text as="div" id="model-label" size="2" weight="medium" mb="2">Model</Text>
              <Select.Root
                value={selected?.id}
                onValueChange={(id) => navigate('match', { ...route.params, model: id })}
              >
                {/* Naming the trigger by the visible label plus its own contents is what
                    makes it announce as "Model, MiniLM-L6" rather than "combobox". */}
                <Select.Trigger
                  id="model-trigger"
                  aria-labelledby="model-label model-trigger"
                  aria-describedby="model-note"
                  style={{ width: '100%' }}
                />
                <Select.Content position="popper">
                  {MODELS.map((m) => (
                    <Select.Item key={m.id} value={m.id}>
                      {m.label} | {m.size}
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Root>
              <Text as="div" id="model-note" size="1" color="gray" mt="2">
                {selected?.note}
              </Text>
            </Box>

            <Box style={{ flex: '0 1 240px', minWidth: 0 }}>
              <Text as="label" htmlFor="threshold" size="2" weight="medium" mb="2"
                style={{ display: 'block' }}>
                Minimum score
              </Text>
              <TextField.Root
                id="threshold"
                aria-describedby="threshold-help"
                type="number"
                min={0}
                max={1}
                step={0.05}
                placeholder="none"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                className="tabular"
              />
              {/* This was a hover tooltip on a non-focusable icon, which put the one
                  sentence that explains the control out of reach of a keyboard entirely.
                  For an audience meeting "minimum score" for the first time, it belongs
                  on the page. */}
              <Text as="div" id="threshold-help" size="1" color="gray" mt="2">
                Entries scoring below this are reported as no match.
              </Text>
              <Flex gap="3" mt="2" wrap="wrap">
                <Button size="1" variant="ghost" color="gray"
                  onClick={() => setThreshold(String(DEFAULT_THRESHOLD))}>
                  Reset to {DEFAULT_THRESHOLD}
                </Button>
                <Button size="1" variant="ghost" color="gray"
                  onClick={() => setThreshold('')}>
                  No minimum
                </Button>
              </Flex>
            </Box>
          </Flex>

          {/* The action sits on its own line rather than inline with the fields. The two
              field boxes are now different heights, so aligning a third column against
              them meant a hand-picked top offset that drifted at every wrap point, and
              one unmistakable primary action reads better here than three peers. */}
          <Flex gap="2" align="center" wrap="wrap">
            <Button size="3" disabled={!ready} onClick={run}>
              <PlayIcon aria-hidden /> Run {selected?.label}
            </Button>
            <Tooltip content={copied ? 'Link copied' : 'Copy a link to this exact setup'}>
              <IconButton size="3" variant="soft" color="gray" onClick={copyLink}
                aria-label="Copy link to this model">
                {copied ? <CheckIcon /> : <Link2Icon />}
              </IconButton>
            </Tooltip>
          </Flex>

          {queries.length === 0 || references.length === 0 ? (
            <Callout.Root color="gray" variant="surface" size="1">
              <Callout.Icon><InfoCircledIcon /></Callout.Icon>
              <Callout.Text>
                Add both lists above, then run. Not sure which model to pick?{' '}
                <Button
                  variant="ghost" size="1" className="inline-action"
                  onClick={() => navigate('choose')}
                >
                  Compare them on your data
                </Button>.
              </Callout.Text>
            </Callout.Root>
          ) : (
            <Text size="1" color="gray" className="tabular">
              {entryLabel(queries.length)} against{' '}
              {references.length.toLocaleString()} references. Everything runs on your
              machine; your data is never uploaded.
            </Text>
          )}
        </Flex>
      </Card>

      {result && <MatchResults result={result} />}
    </Flex>
  );
}

const PREVIEW_ROWS = 200;

/**
 * Memoised below. `result` changes only when a run finishes, but the shell re-renders
 * on every progress tick.
 */
function MatchResultsView({ result }) {
  const { matches, model, threshold } = result;

  // null is the review order, corrected first, which is not any column's sort
  const [sort, setSort] = useState(null);

  // depends only on the result set, not on progress ticks
  const { ordered, changed, unmatched, unchanged } = useMemo(() => {
    const corrected = [];
    const missed = [];
    const kept = [];
    for (const m of matches) {
      if (!m.alethia_prediction) missed.push(m);
      else if (m.alethia_prediction === m.given_entity) kept.push(m);
      else corrected.push(m);
    }
    // corrected rows lead; counts, not arrays, so call sites need not remember which
    return {
      ordered: [...corrected, ...missed, ...kept],
      changed: corrected.length,
      unmatched: missed.length,
      unchanged: kept.length,
    };
  }, [matches]);

  // sort the whole set then slice; sorting the visible 200 answers a different question
  const rows = useMemo(() => sortRows(ordered, sort), [ordered, sort]);

  const toggleSort = (key) => setSort((prev) => cycleSort(prev, key));

  return (
    <Card size="3" variant="surface">
      <Flex direction="column" gap="4">
        <Flex justify="between" align="baseline" gap="3" wrap="wrap">
          <Heading as="h2" size="4" weight="medium">Results</Heading>
          <Text size="1" color="gray">matched with {model}</Text>
        </Flex>

        <Flex gap="5" wrap="wrap">
          <Stat label="entries" value={matches.length} />
          <Stat label="corrected" value={changed} color="amber" />
          <Stat label="already correct" value={unchanged} color="green" />
          {unmatched > 0 && (
            <Stat label="no confident match" value={unmatched} color="red" />
          )}
        </Flex>

        {unmatched > 0 && (
          <Callout.Root color="gray" variant="surface" size="1">
            <Callout.Icon><InfoCircledIcon /></Callout.Icon>
            <Callout.Text>
              No confident match means nothing scored above your minimum. Lower it, or
              clear it to always take the best guess.
            </Callout.Text>
          </Callout.Root>
        )}

        {/* Table.Root already wraps its table in a ScrollArea. The extra vertical-only
            ScrollArea that used to sit here nested a second scroller inside the first and
            then forbade the horizontal axis the inner one needed on a narrow screen.
            Letting the page scroll instead also keeps browser find-in-page working across
            the whole result set, which a 420px window quietly broke. */}
        <Table.Root variant="surface" size="2">
          <Table.Header>
            <Table.Row>
              <SortableHeader
                label="Your entry" sortKey="given" sort={sort} onSort={toggleSort}
              />
              <SortableHeader
                label="Matched to" sortKey="prediction" sort={sort} onSort={toggleSort}
              />
              <SortableHeader
                label="Score" sortKey="score" sort={sort} onSort={toggleSort} align="right"
              />
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {rows.slice(0, PREVIEW_ROWS).map((m, i) => (
              <Table.Row key={`${m.given_entity}-${i}`}>
                <Table.RowHeaderCell style={{ wordBreak: 'break-word' }}>
                  {m.given_entity}
                </Table.RowHeaderCell>
                <Table.Cell style={{ wordBreak: 'break-word' }}>
                  {m.alethia_prediction ?? (
                    <Text color="gray" style={{ fontStyle: 'italic' }}>no match</Text>
                  )}
                </Table.Cell>
                <Table.Cell align="right">
                  {m.alethia_score === null ? (
                    <Text color="gray">-</Text>
                  ) : (
                    <Badge color={scoreColor(m.alethia_score, threshold)} variant="soft"
                      radius="full" className="tabular">
                      {m.alethia_score.toFixed(3)}
                    </Badge>
                  )}
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table.Root>

        <Flex justify="between" align="center" gap="3" wrap="wrap">
          {/* Always rendered, both because "you are seeing everything" is worth saying
              and because justify="between" with a single child silently left-aligns the
              download button whenever the result set was short. */}
          <Flex align="center" gap="2" wrap="wrap">
            <Text size="1" color="gray" className="tabular">
              {matches.length > PREVIEW_ROWS
                ? `Showing the first ${PREVIEW_ROWS} of ${matches.length.toLocaleString()}.`
                : `Showing all ${matches.length.toLocaleString()}.`}
            </Text>
            {/* Which order the rows are in is not obvious from a screenful of them,
                least of all when only the first 200 of several thousand are shown. */}
            <Text size="1" color="gray">
              {sort ? (
                <>
                  Sorted by {SORT_LABELS[sort.key]},{' '}
                  {sort.dir === 'asc' ? 'ascending' : 'descending'}.{' '}
                  <Button
                    variant="ghost" size="1" className="inline-action"
                    onClick={() => setSort(null)}
                  >
                    Back to review order
                  </Button>
                </>
              ) : (
                'Corrected entries first, then anything unmatched.'
              )}
            </Text>
          </Flex>
          {/* `rows`, not `matches`: the file follows the order on screen. Someone who has
              just sorted by score to find the weak matches expects them at the top of the
              download too, not scattered back through the original input order. */}
          <Button variant="soft" onClick={() => downloadCsv(rows, 'alethia-matches.csv')}>
            <DownloadIcon aria-hidden /> Download all {matches.length.toLocaleString()} rows
          </Button>
        </Flex>
      </Flex>
    </Card>
  );
}

const MatchResults = memo(MatchResultsView);

/**
 * A column header that sorts, and says so.
 *
 * A bare th is not focusable and announces as a column name, so the sort is a real
 * button inside it. `aria-sort` conveys the direction and belongs on the th, so both
 * elements are needed.
 */
function SortableHeader({ label, sortKey, sort, onSort, align }) {
  const active = sort?.key === sortKey;
  const direction = active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none';

  return (
    <Table.ColumnHeaderCell align={align} aria-sort={direction}>
      <Button
        variant="ghost"
        color="gray"
        size="1"
        onClick={() => onSort(sortKey)}
        // the caret is not legible to a screen reader, so spell out action and state
        aria-label={
          active
            ? `${label}, sorted ${direction}. Activate to ${
              sort.dir === 'asc' ? 'reverse' : 'return to review order'}.`
            : `${label}, not sorted. Activate to sort.`
        }
      >
        <Flex align="center" gap="1">
          {align === 'right' && <SortCaret active={active} dir={sort?.dir} />}
          <Text weight="bold" size="1">{label}</Text>
          {align !== 'right' && <SortCaret active={active} dir={sort?.dir} />}
        </Flex>
      </Button>
    </Table.ColumnHeaderCell>
  );
}

/** Dimmed both ways when inactive, so every column reads as sortable before it is used. */
function SortCaret({ active, dir }) {
  if (!active) return <CaretSortIcon aria-hidden style={{ opacity: 0.45 }} />;
  return dir === 'asc'
    ? <ArrowUpIcon aria-hidden color="var(--accent-11)" />
    : <ArrowDownIcon aria-hidden color="var(--accent-11)" />;
}

function Stat({ label, value, color = 'gray' }) {
  return (
    <Flex direction="column" gap="1">
      <Text size="6" weight="bold" className="tabular"
        color={color === 'gray' ? undefined : color}>
        {value.toLocaleString()}
      </Text>
      <Text size="1" color="gray">{label}</Text>
    </Flex>
  );
}

/**
 * Colour has to agree with the cutoff the user actually set.
 *
 * These bands were pinned at 0.9 and 0.75 while the default minimum is 0.7, so a match at
 * 0.72, one the user's own threshold had just accepted, was painted the same red this
 * interface uses for "no confident match". Red now means only "below the cutoff", which
 * can appear solely when the cutoff was cleared.
 */
function scoreColor(score, threshold) {
  if (score >= 0.9) return 'green';
  return score >= (threshold ?? DEFAULT_THRESHOLD) ? 'amber' : 'red';
}
