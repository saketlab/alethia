import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge, Box, Button, Callout, Card, Container, Flex, Heading, Progress, Section,
  SegmentedControl, Separator, Tabs, Text,
} from '@radix-ui/themes';
import {
  CrossCircledIcon, DownloadIcon, FileTextIcon, InfoCircledIcon, MagicWandIcon,
  ReloadIcon,
} from '@radix-ui/react-icons';

import ChooseView from './ChooseView.jsx';
import MatchView from './MatchView.jsx';
import { ListInput } from './ListInput.jsx';
import { LiveRegion } from './LiveRegion.jsx';
import { EXAMPLES, EXAMPLE_LIST, DEFAULT_EXAMPLE, matchExample } from './example.js';
import { resolveModel } from './models.js';
import { parseList } from './io.js';
import { useRoute } from './router.js';

const SEEN_KEY = 'alethia.seen';

/**
 * The progress-card label and the screen-reader sentence, formed together because they
 * are the same fact at two volumes.
 */
function describeStage({ step, model }) {
  return step === 'download'
    ? { label: `Downloading ${model}`, announcement: `Downloading ${model}. First run only.` }
    : { label: `Encoding with ${model}`, announcement: `Encoding your entries with ${model}.` };
}

/**
 * Has this browser been here before? Answers "no" on failure; storage is often blocked
 * on managed machines, and a first-timer shown two empty boxes learns nothing.
 */
function firstVisit() {
  try {
    return !window.localStorage.getItem(SEEN_KEY);
  } catch {
    return true;
  }
}

/**
 * Two jobs, two views, two URLs:
 *
 *   #/match?model=<id>   run one model over your lists
 *   #/choose             compare models and get a recommendation
 *
 * One primary action per view, progress reported as real stages, and results that
 * replace the form.
 */
export default function App({ appearance }) {
  const [route, navigate] = useRoute();

  // decided once per mount, so a re-render cannot refill emptied boxes
  const [opensWithExample] = useState(firstVisit);

  // The lists live in the shell so switching views does not lose typed data.
  const [messy, setMessy] = useState(
    () => (opensWithExample ? EXAMPLES[DEFAULT_EXAMPLE].messy : ''),
  );
  const [reference, setReference] = useState(
    () => (opensWithExample ? EXAMPLES[DEFAULT_EXAMPLE].reference : ''),
  );

  // derived, not tracked; a flag would need clearing in every list handler
  const exampleKey = matchExample(messy, reference);
  const isEmpty = messy === '' && reference === '';

  // written on mount, not on edit, so it costs one write per visit
  useEffect(() => {
    try {
      window.localStorage.setItem(SEEN_KEY, '1');
    } catch {
      // Blocked storage only costs the example reappearing next time.
    }
  }, []);

  const clearLists = useCallback(() => {
    setMessy('');
    setReference('');
  }, []);

  const loadExample = useCallback((key = DEFAULT_EXAMPLE) => {
    const picked = EXAMPLES[key] ?? EXAMPLES[DEFAULT_EXAMPLE];
    setMessy(picked.messy);
    setReference(picked.reference);
  }, []);

  const [stage, setStage] = useState(null);
  const [error, setError] = useState(null);
  // stage boundaries only; a live region firing ten times a second is unusable
  const [announcement, setAnnouncement] = useState('');
  const workerRef = useRef(null);

  const queries = useMemo(() => parseList(messy), [messy]);
  const references = useMemo(() => parseList(reference), [reference]);

  // chunk updates re-render dozens of times a second; coalesce, then flush the last
  const progressRef = useRef({ pending: null, timer: null });

  const queueProgress = useCallback((patch) => {
    const p = progressRef.current;
    p.pending = patch;
    p.timer ??= setTimeout(() => {
      p.timer = null;
      const next = p.pending;
      p.pending = null;
      // unchanged when null, so a late flush cannot resurrect the progress card
      setStage((s) => (s ? { ...s, ...next } : s));
    }, 100);
  }, []);

  const stopProgress = useCallback(() => {
    const p = progressRef.current;
    if (p.timer) clearTimeout(p.timer);
    p.timer = null;
    p.pending = null;
  }, []);

  /**
   * One worker for the whole session, shared by the warm-up and every run; an extractor
   * dies with its worker, so a second would re-pay the ONNX session and wasm compile.
   * Cancelling terminates it, so the ref is nulled on the way out.
   */
  const getWorker = useCallback(() => {
    workerRef.current ??= new Worker(
      new URL('../worker.js', import.meta.url), { type: 'module' },
    );
    return workerRef.current;
  }, []);

  const destroyWorker = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
  }, []);

  // a cancel must settle the awaited promise, or the suspended frame leaks its arrays
  const rejectRef = useRef(null);

  const cancel = useCallback(() => {
    stopProgress();
    destroyWorker();
    rejectRef.current?.(new Error('Run cancelled'));
    rejectRef.current = null;
    setStage(null);
    setAnnouncement('Run cancelled.');
  }, [stopProgress, destroyWorker]);

  // a tab closed mid-run would otherwise leave the worker holding a pending encode
  useEffect(() => () => {
    stopProgress();
    destroyWorker();
  }, [stopProgress, destroyWorker]);

  /**
   * Warm the model on load, so the first Run does not wait on the download.
   *
   * Silent, and obeys Data Saver since the audience is often on a metered connection.
   * Never cancelled: a starting run awaits the same in-flight load, and terminating it
   * would discard a partial download Cache Storage does not persist.
   */
  useEffect(() => {
    if (typeof Worker === 'undefined') return; // jsdom, and any SSR-ish host
    const conn = navigator.connection;
    if (conn?.saveData) return;
    if (/(^|-)2g$/.test(conn?.effectiveType ?? '')) return;
    // Warm whatever the page actually opened with, which is not always the default.
    getWorker().postMessage({
      type: 'prefetch', payload: { model: resolveModel(route.params.model) },
    });
    // Mount only. Re-warming on every model change would spend far more than it saves.
  }, [getWorker]);

  /** Start a worker job; resolves with the result payload. */
  const runJob = useCallback((message) => new Promise((resolve, reject) => {
    setError(null);
    setStage({ step: 'starting', label: 'Preparing' });
    setAnnouncement('Starting.');
    stopProgress();
    rejectRef.current = reject;

    const worker = getWorker();

    worker.onmessage = ({ data }) => {
      const { type, payload } = data;
      if (type === 'stage') {
        // drop queued chunk updates; they describe a finished download
        stopProgress();
        const { label, announcement: said } = describeStage(payload);
        setStage({
          step: payload.step,
          label,
          modelIndex: payload.modelIndex,
          modelCount: payload.modelCount,
        });
        setAnnouncement(said);
      } else if (type === 'download') {
        queueProgress({
          detail: `${(payload.loaded / 1e6).toFixed(1)} of ${(payload.total / 1e6).toFixed(1)} MB`,
          percent: (payload.loaded / payload.total) * 100,
        });
      } else if (type === 'encode') {
        queueProgress({
          detail: `${payload.done.toLocaleString()} of ${payload.total.toLocaleString()} ${payload.side}`,
          percent: (payload.done / payload.total) * 100,
        });
      } else if (type === 'result') {
        stopProgress();
        rejectRef.current = null;
        setStage(null);
        setAnnouncement('Finished. Your results are below.');
        resolve(payload);
      } else if (type === 'error') {
        stopProgress();
        rejectRef.current = null;
        setStage(null);
        setAnnouncement('');
        setError(payload.message);
        reject(new Error(payload.message));
      }
      // prefetched can arrive late and must not settle the job; only result may
    };

    worker.postMessage(message);
  }), [queueProgress, stopProgress, getWorker]);

  // a button that swaps views unmounts itself, dropping focus to <body>; a tab click
  // does not, so only take focus in the first case
  const viewRef = useRef(null);
  const lastView = useRef(route.view);
  useEffect(() => {
    if (lastView.current === route.view) return;
    lastView.current = route.view;
    const active = document.activeElement;
    if (!active || active === document.body) viewRef.current?.focus();
  }, [route.view]);

  // a new props object per progress tick re-renders the 200-row table for nothing
  const busy = Boolean(stage);
  const shared = useMemo(
    () => ({ queries, references, runJob, busy, navigate }),
    [queries, references, runJob, busy, navigate],
  );

  return (
    <Container size="4" px={{ initial: '4', sm: '6' }} py="6">
      <Button asChild size="2" className="skip-link">
        <a href="#main-content">Skip to your lists</a>
      </Button>

      <Box asChild>
        <header>
          <Header appearance={appearance} />
        </header>
      </Box>

      <LiveRegion>{announcement}</LiveRegion>

      <main id="main-content" tabIndex={-1}>
        {exampleKey && (
          <Section size="1" pt="4" pb="0">
            <ExampleNotice
              view={route.view}
              current={exampleKey}
              onPick={loadExample}
              onClear={clearLists}
            />
          </Section>
        )}

        <Section size="2" pt="4" pb="0">
          <ListInputs
            messy={messy}
            reference={reference}
            onMessy={setMessy}
            onReference={setReference}
            queries={queries}
            references={references}
          />
          {/* beside the boxes it fills, so neither view renders it under its own rule */}
          {isEmpty && (
            <Flex pt="4" direction="column" gap="2" align="start">
              <Text size="2" color="gray">
                Nothing to match yet. Load a worked example to see how it works.
              </Text>
              {/* wrapped: onClick would otherwise hand the handler a click event */}
              <Button size="2" variant="solid" onClick={() => loadExample()}>
                <MagicWandIcon aria-hidden /> Load the example
              </Button>
            </Flex>
          )}
        </Section>

        <Section size="2" pt="5" pb="0">
          <Tabs.Root
            value={route.view}
            onValueChange={(view) => navigate(view, view === 'match' ? route.params : {})}
          >
            <Tabs.List size="2">
              <Tabs.Trigger value="match">Match</Tabs.Trigger>
              <Tabs.Trigger value="choose">Choose a model</Tabs.Trigger>
            </Tabs.List>

            <Box pt="4">
              <Text size="2" color="gray">
                {route.view === 'match'
                  ? 'Run one model over your lists and get the corrected entries.'
                  : 'Compare several models on your own data and see which suits it best. '
                    + 'No correct answers needed.'}
              </Text>
            </Box>

            <Box pt="4" ref={viewRef} tabIndex={-1} className="view-panel">
              <Tabs.Content value="match">
                <MatchView {...shared} modelId={route.params.model} route={route} />
              </Tabs.Content>
              <Tabs.Content value="choose">
                <ChooseView {...shared} />
              </Tabs.Content>
            </Box>
          </Tabs.Root>
        </Section>

        {stage && <RunProgress stage={stage} onCancel={cancel} />}

        {/* Always mounted for the same reason as the status region above: an alert node
            created together with its text is unreliably announced. */}
        <div role="alert">
          {error && (
            <Section size="2">
              <Callout.Root color="red">
                <Callout.Icon><CrossCircledIcon /></Callout.Icon>
                <Callout.Text>{error}</Callout.Text>
              </Callout.Root>
            </Section>
          )}
        </div>
      </main>

      <Box asChild>
        <footer>
          <Footer />
        </footer>
      </Box>
    </Container>
  );
}

/**
 * What is already in the boxes, and how to get rid of it.
 *
 * A first visit lands on a working demonstration. This line says plainly that the
 * entries are samples, since sample data mistaken for real data is the worse failure.
 */
function ExampleNotice({ view, current, onPick, onClear }) {
  const example = EXAMPLES[current];
  const counts = useMemo(() => ({
    queries: example.messy.split('\n').length,
    references: example.reference.split('\n').length,
  }), [example]);

  return (
    <Card size="2" variant="surface">
      <Flex direction="column" gap="3">
        <Flex align="center" gap="2" wrap="wrap">
          <FileTextIcon aria-hidden />
          <Text size="2" weight="medium">Example data</Text>
          <Badge color="gray" variant="soft" radius="full" className="tabular">
            {counts.queries} entries against {counts.references} references
          </Badge>
        </Flex>

        <Text size="2" color="gray" style={{ maxWidth: '70ch' }}>
          {example.blurb}{' '}
          {view === 'match'
            ? 'Run it to see them corrected, or'
            : 'Compare a few models to see which reads them best, or'}{' '}
          <Button variant="ghost" size="1" className="inline-action" onClick={onClear}>
            clear it and use your own data
          </Button>.
        </Text>

        {/* One example only ever showed the easy case, which is the least convincing
            thing this tool can demonstrate: typos are solvable by string similarity, so a
            visitor who saw only that had no reason to believe the embedding models earn
            their download. These are ordered by difficulty for the same reason, running
            the first and the last is the fastest way to see what the choice costs. */}
        <Flex direction="column" gap="2">
          <Text size="1" color="gray">
            Try another kind of list. They get harder from left to right.
          </Text>
          <Flex gap="2" wrap="wrap" role="group" aria-label="Example dataset">
            {EXAMPLE_LIST.map((e) => {
              const active = e.key === current;
              return (
                <Button
                  key={e.key}
                  size="1"
                  radius="full"
                  variant={active ? 'solid' : 'soft'}
                  color={active ? undefined : 'gray'}
                  // aria-pressed, not colour, tells a screen reader which is loaded
                  aria-pressed={active}
                  onClick={() => onPick(e.key)}
                >
                  {e.label}
                </Button>
              );
            })}
          </Flex>
        </Flex>
      </Flex>
    </Card>
  );
}

// transformers.js keeps models in Cache Storage, which exists only in a secure context.
// Over plain http on a LAN address every reload downloads the model again.
const cachesModels = () =>
  typeof window === 'undefined' || window.isSecureContext !== false;

function Footer() {
  return (
    <Section size="1" pt="6" pb="2">
      <Separator size="4" mb="4" />
      <Text size="1" color="gray">
        Your lists never leave this device, and are therefore safe. Models are
        downloaded from Hugging Face and run in your browser.
        {!cachesModels() && (
          <>
            {' '}This page is not on a secure origin, so the browser cannot store the
            model and each reload downloads it again. Use localhost or https to keep it.
          </>
        )}
      </Text>
    </Section>
  );
}

function Header({ appearance }) {
  return (
    <Flex direction="column" gap="2">
      <Flex align="center" gap="3" justify="between" wrap="wrap">
        <Flex align="center" gap="3">
          <Heading as="h1" size="7" weight="bold" trim="start">alethia</Heading>
          <Badge color="gray" variant="soft" radius="full">runs in your browser</Badge>
        </Flex>
        {appearance && (
          <SegmentedControl.Root
            size="1"
            value={appearance.preference}
            onValueChange={appearance.choose}
            aria-label="Colour scheme"
          >
            <SegmentedControl.Item value="system">System</SegmentedControl.Item>
            <SegmentedControl.Item value="light">Light</SegmentedControl.Item>
            <SegmentedControl.Item value="dark">Dark</SegmentedControl.Item>
          </SegmentedControl.Root>
        )}
      </Flex>
      <Text size="3" color="gray" style={{ maxWidth: '58ch' }}>
        Clean up a messy list of names, and find which embedding model suits your
        data, measured on your own entries rather than a leaderboard.
      </Text>
    </Flex>
  );
}

function ListInputs({ messy, reference, onMessy, onReference, queries, references }) {
  return (
    <Flex direction={{ initial: 'column', md: 'row' }} gap="5">
      {/* minWidth 0 lets a flex child shrink below its content width; without it a single
          very long pasted name would push the second list off the page. */}
      <Box style={{ flex: 1, minWidth: 0 }}>
        <ListInput
          title="Messy entries"
          hint="The names you want cleaned up. One per line."
          placeholder={'Bombay\nCalcutta\nMadras'}
          value={messy}
          onChange={onMessy}
          count={queries.length}
        />
      </Box>
      <Box style={{ flex: 1, minWidth: 0 }}>
        <ListInput
          title="Correct entries"
          hint="The canonical list to match against. One per line."
          placeholder={'Mumbai\nKolkata\nChennai'}
          value={reference}
          onChange={onReference}
          count={references.length}
        />
      </Box>
    </Flex>
  );
}

function RunProgress({ stage, onCancel }) {
  const isDownload = stage.step === 'download';
  return (
    <Section size="2" pt="5">
      <Card size="4" variant="surface" aria-busy="true">
        <Flex direction="column" gap="4">
          <Flex align="center" gap="3">
            {isDownload
              ? <DownloadIcon aria-hidden />
              : <ReloadIcon className="spin" aria-hidden />}
            <Heading as="h2" size="4" weight="medium">{stage.label}</Heading>
          </Flex>
          <Progress value={stage.percent ?? undefined} size="3" aria-label={stage.label} />
          <Flex justify="between" align="center" gap="3" wrap="wrap">
            <Text size="2" color="gray" className="tabular">
              {stage.detail ?? 'Working...'}
              {stage.modelCount > 1
                && ` | model ${stage.modelIndex + 1} of ${stage.modelCount}`}
            </Text>
            <Button variant="soft" color="gray" onClick={onCancel}>Cancel</Button>
          </Flex>
          {isDownload && (
            <Callout.Root color="gray" variant="surface" size="1">
              <Callout.Icon><InfoCircledIcon /></Callout.Icon>
              <Callout.Text>
                First run only. Your browser caches the model afterwards.
              </Callout.Text>
            </Callout.Root>
          )}
        </Flex>
      </Card>
    </Section>
  );
}
