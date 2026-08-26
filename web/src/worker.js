// Embedding and scoring, off the main thread; every message is progress or a result.

import { pipeline, env } from '@huggingface/transformers';

import { clusterEntities, fromRows, matchByEmbeddings } from './core.js';
import {
  centeredSeparability,
  hubness,
  intrinsicDimensionality,
  mutualNnRate,
  referenceSeparability,
  retrievalMargin,
  uniformityLoss,
} from './metrics.js';
import { compositeScores } from './assess.js';

// the Hugging Face CDN pays the bandwidth and serves from an edge near the user
env.allowLocalModels = false;

// the promise is cached, not the value; caching after the await builds two pipelines
const extractors = new Map();
// a run sharing an in-flight warm-up must subscribe to a load someone else started
const progressListeners = new Map();

function getExtractor(modelId, onProgress) {
  let listeners = progressListeners.get(modelId);
  if (!listeners) {
    listeners = new Set();
    progressListeners.set(modelId, listeners);
  }
  if (onProgress) listeners.add(onProgress);

  let pending = extractors.get(modelId);
  if (!pending) {
    pending = pipeline('feature-extraction', modelId, {
      dtype: 'q8', // int8: ~23 MB instead of ~90 MB, within a few % of full precision
      progress_callback: (p) => {
        if (p.status === 'progress' && p.total) {
          for (const fn of listeners) fn({ file: p.file, loaded: p.loaded, total: p.total });
        }
      },
    });
    extractors.set(modelId, pending);
    // A failed load must not stay cached, or every retry replays the same rejection.
    pending.catch(() => extractors.delete(modelId));
  }
  return pending.finally(() => listeners.delete(onProgress));
}

/**
 * Encode in batches, reporting progress between them.
 *
 * Mean pooling plus L2 normalisation matches sentence-transformers, so these numbers
 * agree with the Python and R packages. Getting it wrong is silent: SapBERT carries its
 * representation on [CLS] and mean-pools to plausible embeddings that rank worse, so
 * each model declares its own pooling in models.js.
 */
async function encode(extractor, texts, onProgress, pooling = 'mean', batchSize = 64) {
  const rows = [];
  for (let start = 0; start < texts.length; start += batchSize) {
    const chunk = texts.slice(start, start + batchSize);
    const out = await extractor(chunk, { pooling, normalize: true });
    const [n, d] = out.dims;
    for (let i = 0; i < n; i++) {
      rows.push(out.data.slice(i * d, (i + 1) * d));
    }
    onProgress(Math.min(start + batchSize, texts.length), texts.length);
    // Yield so a cancel message can be delivered between batches.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return fromRows(rows);
}

function metricsFor(queryEmb, refEmb) {
  return {
    ...referenceSeparability(refEmb),
    ...centeredSeparability(refEmb),
    ...intrinsicDimensionality(refEmb),
    ...retrievalMargin(queryEmb, refEmb),
    ...hubness(queryEmb, refEmb, 5),
    uniformity_loss: uniformityLoss(refEmb),
    mutual_nn_rate: mutualNnRate(queryEmb, refEmb, 5),
  };
}

const post = (t, p) => self.postMessage({ type: t, payload: p });

// Download the model and encode one or both sides, reporting each stage as it starts.
// Every job needs this sequence; only the sides differ.
async function prepare(model, sides, modelIndex = 0, modelCount = 1) {
  const stage = (step) =>
    post('stage', { model: model.label, step, modelIndex, modelCount });

  stage('download');
  const extractor = await getExtractor(model.id, (p) =>
    post('download', { model: model.label, ...p }));

  stage('encode');
  const out = {};
  for (const [side, texts] of Object.entries(sides)) {
    out[side] = await encode(extractor, texts, (done, total) =>
      post('encode', { model: model.label, done, total, side }),
    model.pooling);
  }
  return out;
}

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  try {
    // warm the cache silently, so the first Run does not wait on the download
    if (type === 'prefetch') {
      await getExtractor(payload.model.id, () => {});
      post('prefetched', { model: payload.model.label });
      return;
    }

    if (type === 'assess') {
      const { queries, references, models } = payload;
      const assessments = [];

      for (const [index, model] of models.entries()) {
        const { references: refEmb, queries: queryEmb } = await prepare(
          model, { references, queries }, index, models.length,
        );

        assessments.push({
          model: model.label,
          metrics: metricsFor(queryEmb, refEmb),
          embeddings: { queryEmb, refEmb },
        });
      }

      // Composite ranking across models, as the Python and R packages compute it.
      const scored = compositeScores(
        assessments.map(({ model, metrics }) => ({ model, metrics, error: null })),
      );
      const best = scored.reduce((a, b) => (b.score > a.score ? b : a));
      const winner = assessments.find((a) => a.model === best.model);

      const matches = matchByEmbeddings(
        queries, references, winner.embeddings.queryEmb, winner.embeddings.refEmb,
      );

      post('result', {
        kind: 'assess',
        ranking: scored,
        best: best.model,
        bestId: models.find((m) => m.label === best.model)?.id,
        matches,
        metrics: Object.fromEntries(assessments.map((a) => [a.model, a.metrics])),
      });
      return;
    }

    if (type === 'match') {
      const { queries, references, model, threshold } = payload;
      // carry the cutoff back, so the table cannot mislabel it
      const cutoff = Number.isFinite(threshold) ? threshold : null;
      const { references: refEmb, queries: queryEmb } = await prepare(
        model, { references, queries },
      );

      post('result', {
        kind: 'match',
        model: model.label,
        threshold: cutoff,
        matches: matchByEmbeddings(queries, references, queryEmb, refEmb, cutoff),
      });
      return;
    }

    if (type === 'cluster') {
      const { entities, model, floor } = payload;
      const unique = [...new Set(entities)];
      const { entities: emb } = await prepare(model, { entities: unique });

      const result = clusterEntities(unique, emb, { floor });
      post('result', {
        clusters: result.entities.map((entity, i) => ({
          entity,
          cluster: result.labels[i],
          canonical: result.canonical.get(result.labels[i]),
        })),
        edges: result.edges.length,
      });
      return;
    }

    throw new Error(`unknown message type: ${type}`);
  } catch (error) {
    post('error', { message: error?.message ?? String(error) });
  }
};
