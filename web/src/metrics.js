// Label-free geometric metrics, ported from alethia.assess.metrics. Each states its
// formula so the three implementations can be checked against one another.

import { l2Normalize, mean, median, popStd, rankKey, topK } from './core.js';

function simsRow(x, i, y) {
  const out = new Float64Array(y.rows);
  const xo = i * x.cols;
  for (let j = 0; j < y.rows; j++) {
    const yo = j * y.cols;
    let sum = 0;
    for (let k = 0; k < x.cols; k++) sum += x.data[xo + k] * y.data[yo + k];
    out[j] = sum;
  }
  return out;
}

/**
 * How well distinct references are kept apart: cosine to the nearest *other* reference.
 *
 * Raw cosines, so they partly reflect anisotropy: transformer embeddings tend to occupy
 * a narrow cone that pushes every cosine high. See centeredSeparability for the
 * corrected view.
 */
export function referenceSeparability(refEmb, confusionThreshold = 0.9) {
  const x = l2Normalize(refEmb);
  const n = x.rows;
  if (n < 2) {
    return { mean_nn_similarity: 0, median_nn_similarity: 0, confusability_rate: 0 };
  }
  const nn = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = simsRow(x, i, x);
    s[i] = -Infinity;
    let best = -Infinity;
    for (let j = 0; j < n; j++) if (s[j] > best) best = s[j];
    nn[i] = best;
  }
  let confusable = 0;
  for (const v of nn) if (v > confusionThreshold) confusable++;
  return {
    mean_nn_similarity: mean(nn),
    median_nn_similarity: median(nn),
    confusability_rate: confusable / n,
  };
}

/** Separability after removing the set mean, which strips the shared cone direction. */
export function centeredSeparability(refEmb) {
  const n = refEmb.rows;
  const d = refEmb.cols;
  if (n < 3) return { centered_nn_similarity: 0, nn_margin_z: 0 };

  const centered = new Float64Array(refEmb.data.length);
  for (let j = 0; j < d; j++) {
    let mu = 0;
    for (let i = 0; i < n; i++) mu += refEmb.data[i * d + j];
    mu /= n;
    for (let i = 0; i < n; i++) centered[i * d + j] = refEmb.data[i * d + j] - mu;
  }
  const x = l2Normalize({ data: centered, rows: n, cols: d });

  const nn = new Float64Array(n);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = simsRow(x, i, x);
    // Over the n-1 other references; a reference is not its own neighbour.
    const others = new Float64Array(n - 1);
    let w = 0;
    let best = -Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      others[w++] = s[j];
      if (s[j] > best) best = s[j];
    }
    nn[i] = best;
    const mu = mean(others);
    const sd = popStd(others);
    // A degenerate row has no scale to express a margin in, so it drops out.
    z[i] = sd > 0 ? (best - mu) / sd : NaN;
  }
  const defined = Array.from(z).filter((v) => !Number.isNaN(v));
  return {
    centered_nn_similarity: mean(nn),
    nn_margin_z: defined.length ? mean(defined) : NaN,
  };
}

/**
 * Eigenvalues of a symmetric matrix by the cyclic Jacobi method.
 *
 * Runs only on the d-by-d feature covariance, never the n-by-n similarity matrix, so it
 * stays small however much data the user pastes in.
 */
export function symmetricEigenvalues(a, d, { sweeps = 60, tol = 1e-12 } = {}) {
  const m = Float64Array.from(a);
  const at = (i, j) => m[i * d + j];
  const set = (i, j, v) => {
    m[i * d + j] = v;
  };
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) off += at(p, q) ** 2;
    }
    if (off <= tol) break;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = at(p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (at(q, q) - at(p, p)) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = at(k, p);
          const akq = at(k, q);
          set(k, p, c * akp - s * akq);
          set(k, q, s * akp + c * akq);
        }
        for (let k = 0; k < d; k++) {
          const apk = at(p, k);
          const aqk = at(q, k);
          set(p, k, c * apk - s * aqk);
          set(q, k, s * apk + c * aqk);
        }
      }
    }
  }
  const eigenvalues = new Float64Array(d);
  for (let i = 0; i < d; i++) eigenvalues[i] = at(i, i);
  return eigenvalues;
}

/**
 * Participation ratio over the PCA eigenvalues, and the same divided by the ambient
 * dimension. Higher means the model spreads data over more directions than it collapses.
 */
export function intrinsicDimensionality(emb) {
  const n = emb.rows;
  const d = emb.cols;
  if (n < 2) return { participation_ratio: 0, normalized_pr: 0 };

  const centered = new Float64Array(emb.data.length);
  for (let j = 0; j < d; j++) {
    let mu = 0;
    for (let i = 0; i < n; i++) mu += emb.data[i * d + j];
    mu /= n;
    for (let i = 0; i < n; i++) centered[i * d + j] = emb.data[i * d + j] - mu;
  }
  // shares the nonzero eigenvalues of the n-by-n Gram matrix and is far smaller
  const cov = new Float64Array(d * d);
  for (let p = 0; p < d; p++) {
    for (let q = p; q < d; q++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += centered[i * d + p] * centered[i * d + q];
      cov[p * d + q] = sum;
      cov[q * d + p] = sum;
    }
  }
  const lam = symmetricEigenvalues(cov, d);
  let total = 0;
  let sqTotal = 0;
  for (let i = 0; i < d; i++) {
    const v = Math.max(lam[i], 0);
    total += v;
    sqTotal += v * v;
  }
  if (total <= 0) return { participation_ratio: 0, normalized_pr: 0 };
  const pr = (total * total) / sqTotal;
  return { participation_ratio: pr, normalized_pr: pr / d };
}

/** Mean distance between matched positive pairs; lower means dirt moves a string less. */
export function alignmentLoss(srcEmb, varEmb, alpha = 2) {
  const a = l2Normalize(srcEmb);
  const b = l2Normalize(varEmb);
  if (!a.rows || a.rows !== b.rows) return NaN;
  const out = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    let sum = 0;
    for (let j = 0; j < a.cols; j++) sum += (a.data[i * a.cols + j] - b.data[i * b.cols + j]) ** 2;
    out[i] = Math.sqrt(Math.max(0, sum)) ** alpha;
  }
  return mean(out);
}

/** log E exp(-t d^2) over all pairs; more negative means a better-spread space. */
export function uniformityLoss(emb, t = 2, maxPoints = 2000) {
  let x = emb;
  let n = x.rows;
  if (n < 2) return NaN;
  if (n > maxPoints) {
    // Even stride, matching the linspace subsample elsewhere.
    const idx = new Int32Array(maxPoints);
    for (let i = 0; i < maxPoints; i++) {
      idx[i] = Math.trunc((i * (n - 1)) / (maxPoints - 1));
    }
    const data = new Float64Array(maxPoints * x.cols);
    for (let i = 0; i < maxPoints; i++) {
      data.set(x.data.subarray(idx[i] * x.cols, (idx[i] + 1) * x.cols), i * x.cols);
    }
    x = { data, rows: maxPoints, cols: x.cols };
    n = maxPoints;
  }
  x = l2Normalize(x);
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < n - 1; i++) {
    const s = simsRow(x, i, x);
    for (let j = i + 1; j < n; j++) {
      // For unit rows ||x-y||^2 = 2 - 2 x.y, so no difference vector is ever formed.
      total += Math.exp(-t * Math.max(0, 2 - 2 * s[j]));
      pairs++;
    }
  }
  return Math.log(total / pairs);
}

/** How decisively the top reference beats the runner-up, raw and in per-query sigmas. */
export function retrievalMargin(queryEmb, refEmb, lowMargin = 0.02) {
  const q = l2Normalize(queryEmb);
  const r = l2Normalize(refEmb);
  if (r.rows < 2) {
    // no runner-up, so no margin; Python reports these constants, not NaN
    return { mean_margin: 0, low_margin_rate: 1, mean_margin_z: 0, low_margin_z_rate: 1 };
  }
  if (!q.rows) {
    return {
      mean_margin: NaN, low_margin_rate: NaN, mean_margin_z: NaN, low_margin_z_rate: NaN,
    };
  }
  const margins = new Float64Array(q.rows);
  const zs = new Float64Array(q.rows);
  for (let i = 0; i < q.rows; i++) {
    const s = simsRow(q, i, r);
    const best = topK(s, 2);
    const margin = s[best[0]] - s[best[1]];
    margins[i] = margin;
    const sd = popStd(s);
    zs[i] = sd > 0 ? margin / sd : NaN;
  }
  let low = 0;
  let lowZ = 0;
  for (let i = 0; i < margins.length; i++) {
    if (margins[i] < lowMargin) low++;
    // NaN < 1 is false, so an undefined z is unknown, not low; it stays in the denominator
    if (zs[i] < 1) lowZ++;
  }
  const definedZ = Array.from(zs).filter((v) => !Number.isNaN(v));
  return {
    mean_margin: mean(margins),
    low_margin_rate: low / margins.length,
    mean_margin_z: definedZ.length ? mean(definedZ) : NaN,
    low_margin_z_rate: lowZ / zs.length,
  };
}

/**
 * Skewness of the k-occurrence distribution: are a few references absorbing most
 * queries? High values are pathological. After Radovanovic et al. (2010).
 */
export function hubness(queryEmb, refEmb, k = 5) {
  const q = l2Normalize(queryEmb);
  const r = l2Normalize(refEmb);
  const nRef = r.rows;
  const kk = Math.min(k, nRef);
  if (kk < 1 || !q.rows) return { hubness_skew: 0 };
  const counts = new Float64Array(nRef);
  for (let i = 0; i < q.rows; i++) {
    for (const j of topK(simsRow(q, i, r), kk)) counts[j]++;
  }
  const mu = mean(counts);
  const sigma = popStd(counts);
  if (sigma === 0) return { hubness_skew: 0 };
  let acc = 0;
  for (const c of counts) acc += ((c - mu) / sigma) ** 3;
  return { hubness_skew: acc / counts.length };
}

/** Fraction of top-1 query-to-reference picks that the reference reciprocates. */
export function mutualNnRate(queryEmb, refEmb, k = 5) {
  const q = l2Normalize(queryEmb);
  const r = l2Normalize(refEmb);
  if (!q.rows || !r.rows) return NaN;
  const kk = Math.min(k, q.rows);
  const top1 = new Int32Array(q.rows);
  for (let i = 0; i < q.rows; i++) top1[i] = topK(simsRow(q, i, r), 1)[0];

  // Only the references actually chosen need their query row computed.
  const chosen = [...new Set(Array.from(top1))];
  const backTopK = new Map();
  for (const ref of chosen) {
    backTopK.set(ref, new Set(topK(simsRow(r, ref, q), kk)));
  }
  let reciprocated = 0;
  for (let i = 0; i < q.rows; i++) {
    if (backTopK.get(top1[i]).has(i)) reciprocated++;
  }
  return reciprocated / q.rows;
}

/**
 * Percentile rank of each synthesized positive among all variants, averaged.
 *
 * `src[i]` and `variant[i]` are a positive pair. Unlike alignmentLoss this uses only
 * the *ordering* of each source's similarities, so it is invariant to any strictly
 * increasing transformation of cosine, including the affine shift from adding a
 * common cone direction and renormalising. That invariance is why it carries the
 * largest single weight in the composite. Ties take their mid-rank.
 */
export function positivePairRank(srcEmb, varEmb) {
  if (!srcEmb.rows || !varEmb.rows) return NaN;
  if (srcEmb.rows !== varEmb.rows) {
    throw new Error('srcEmb and varEmb must have the same number of rows');
  }
  const src = l2Normalize(srcEmb);
  const variant = l2Normalize(varEmb);
  const n = src.rows;
  const ranks = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const sims = simsRow(src, i, variant);
    // an exact == misses equal similarities, shifting mid-ranks between runtimes
    const positive = rankKey(sims[i]);
    let lower = 0;
    let equal = 0;
    for (let j = 0; j < n; j++) {
      const v = rankKey(sims[j]);
      if (v < positive) lower++;
      else if (v === positive) equal++;
    }
    ranks[i] = (lower + 0.5 * equal) / n;
  }
  return mean(ranks);
}
