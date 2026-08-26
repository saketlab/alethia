// Port of alethia.assess.assessor. Metrics are z-scored across models, so the composite
// is meaningless for one model; weights normalise within family so correlated views of
// one property cannot count twice.

import { popStd } from './core.js';

/** +1 where higher is better, -1 where lower is better. */
export const METRIC_ORIENTATION = {
  mean_nn_similarity: -1,
  confusability_rate: -1,
  centered_nn_similarity: -1,
  nn_margin_z: 1,
  normalized_pr: 1,
  alignment_loss: -1,
  positive_pair_rank: 1,
  uniformity_loss: -1,
  mean_margin: 1,
  low_margin_rate: -1,
  mean_margin_z: 1,
  low_margin_z_rate: -1,
  hubness_skew: -1,
  mutual_nn_rate: 1,
};

/** Which latent property each metric probes. */
export const METRIC_FAMILY = {
  mean_nn_similarity: 'separability',
  confusability_rate: 'separability',
  centered_nn_similarity: 'separability',
  nn_margin_z: 'separability',
  normalized_pr: 'geometry',
  uniformity_loss: 'geometry',
  alignment_loss: 'robustness',
  positive_pair_rank: 'robustness',
  mean_margin: 'retrieval',
  low_margin_rate: 'retrieval',
  mean_margin_z: 'retrieval',
  low_margin_z_rate: 'retrieval',
  mutual_nn_rate: 'retrieval',
  hubness_skew: 'pathology',
};

export const FAMILY_WEIGHTS = {
  separability: 2.0, // can the model tell distinct references apart at all
  robustness: 1.5,   // do dirty variants stay near their clean form
  retrieval: 1.5,    // are query-to-reference decisions decisive and reciprocated
  geometry: 1.0,     // is the space well spread rather than collapsed
  pathology: 0.75,   // hubness and similar failure modes
};

/**
 * Default per-metric weights.
 *
 * Raw-cosine and absolute-distance quantities score zero: a common cone direction moves
 * them while preserving every neighbour ordering. The robustness family scores zero too,
 * since this build cannot reproduce CPython's Mersenne Twister variants; family
 * normalisation drops it.
 */
export const DEFAULT_WEIGHTS = {
  mean_nn_similarity: 0.0,
  confusability_rate: 0.0,
  centered_nn_similarity: 1.0,
  nn_margin_z: 0.0,
  normalized_pr: 0.0,
  alignment_loss: 0.0,
  positive_pair_rank: 0.0,
  uniformity_loss: 0.0,
  mean_margin: 0.0,
  low_margin_rate: 0.0,
  mean_margin_z: 1.0,
  low_margin_z_rate: 0.5,
  hubness_skew: 0.75,
  mutual_nn_rate: 1.0,
};

/**
 * Rescale weights so each family contributes at most its family weight.
 *
 * `available` must list only metrics that vary across the models compared; a constant
 * one would absorb family weight it never spends.
 */
export function familyNormalizedWeights(weights, available) {
  const usable = Object.keys(weights).filter(
    (k) => available.has(k) && weights[k] > 0,
  );
  const byFamily = new Map();
  for (const key of usable) {
    const family = METRIC_FAMILY[key] ?? key;
    if (!byFamily.has(family)) byFamily.set(family, []);
    byFamily.get(family).push(key);
  }
  const scaled = {};
  for (const [family, keys] of byFamily) {
    const total = keys.reduce((sum, k) => sum + weights[k], 0);
    if (total <= 0) continue;
    const familyWeight = FAMILY_WEIGHTS[family] ?? 1.0;
    for (const k of keys) scaled[k] = (weights[k] / total) * familyWeight;
  }
  return scaled;
}

/**
 * Score each model, best first.
 *
 * @param {{model: string, metrics: object, error: ?string}[]} assessments
 * @returns {{model: string, score: number}[]} sorted by descending score
 */
export function compositeScores(assessments, weights = DEFAULT_WEIGHTS,
  familyNormalize = true) {
  const valid = assessments.filter((a) => !a.error);
  if (valid.length < 2) {
    return assessments.map((a) => ({ model: a.model, score: NaN }));
  }

  const totals = new Map(valid.map((a) => [a.model, 0]));
  let effective = weights;

  if (familyNormalize) {
    const available = new Set();
    const allKeys = new Set(valid.flatMap((a) => Object.keys(a.metrics)));
    for (const key of allKeys) {
      const finite = valid
        .map((a) => a.metrics[key])
        .filter((v) => Number.isFinite(v));
      if (finite.length >= 2 && popStd(finite) > 0) available.add(key);
    }
    effective = familyNormalizedWeights(weights, available);
  }

  for (const key of Object.keys(effective)) {
    let values = valid.map((a) => a.metrics[key]);
    // Hubness is pathological in either direction; only its magnitude ranks models.
    if (key === 'hubness_skew') values = values.map((v) => Math.abs(v));
    const finiteMask = values.map((v) => Number.isFinite(v));
    if (finiteMask.filter(Boolean).length < 2) continue;

    const orientation = METRIC_ORIENTATION[key] ?? 1;
    const finite = values.filter((_, i) => finiteMask[i]);
    // impute at the worst observed value, so failing cannot beat scoring badly
    const worst = orientation > 0 ? Math.min(...finite) : Math.max(...finite);
    values = values.map((v, i) => (finiteMask[i] ? v : worst));

    const mu = values.reduce((s, v) => s + v, 0) / values.length;
    const sigma = popStd(values);
    if (sigma === 0) continue;

    valid.forEach((a, i) => {
      const z = ((values[i] - mu) / sigma) * orientation * effective[key];
      totals.set(a.model, totals.get(a.model) + z);
    });
  }

  return assessments
    .map((a) => ({ model: a.model, score: a.error ? NaN : totals.get(a.model) }))
    .sort((x, y) => (Number.isNaN(x.score) ? 1 : 0) - (Number.isNaN(y.score) ? 1 : 0)
      || y.score - x.score);
}
