// Core numerics: embeddings in, matches and clusters out. Held flat as
// { data: Float64Array, rows, cols }; wasm32 caps a tab at 4 GiB and 60k small arrays
// spend much of it on object headers.

/**
 * Decimal places similarities are rounded to before any ranking decision.
 *
 * Ties are common in real data, and unrounded they turn on the last bits of a dot
 * product, which differ between BLAS builds and languages. Reported scores stay
 * unrounded.
 */
export const RANK_DECIMALS = 12;

const RANK_SCALE = 10 ** RANK_DECIMALS;

/**
 * Round half to even, matching numpy's and R's rule. Math.round rounds halves up, which
 * would bucket a value on a half differently from the other two implementations.
 */
export function rankKey(value) {
  const scaled = value * RANK_SCALE;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let rounded;
  if (diff > 0.5) rounded = floor + 1;
  else if (diff < 0.5) rounded = floor;
  else rounded = floor % 2 === 0 ? floor : floor + 1;
  return rounded / RANK_SCALE;
}

/** Wrap a flat buffer as a matrix view. */
export function matrix(data, rows, cols) {
  return { data, rows, cols };
}

/** Build a matrix from an array of equal-length rows. */
export function fromRows(rows) {
  const nRows = rows.length;
  const nCols = nRows ? rows[0].length : 0;
  const data = new Float64Array(nRows * nCols);
  for (let i = 0; i < nRows; i++) data.set(rows[i], i * nCols);
  return matrix(data, nRows, nCols);
}

export function row(m, i) {
  return m.data.subarray(i * m.cols, (i + 1) * m.cols);
}

/**
 * Row-wise L2 normalisation. Zero rows stay zero, matching the Python and R packages.
 */
export function l2Normalize(m) {
  const out = new Float64Array(m.data.length);
  for (let i = 0; i < m.rows; i++) {
    const off = i * m.cols;
    let sum = 0;
    for (let j = 0; j < m.cols; j++) sum += m.data[off + j] ** 2;
    const norm = sum > 0 ? Math.sqrt(sum) : 1;
    for (let j = 0; j < m.cols; j++) out[off + j] = m.data[off + j] / norm;
  }
  return matrix(out, m.rows, m.cols);
}

/** Dot product of row `i` of `a` with row `j` of `b`. */
function dot(a, i, b, j) {
  const ao = i * a.cols;
  const bo = j * b.cols;
  let sum = 0;
  for (let k = 0; k < a.cols; k++) sum += a.data[ao + k] * b.data[bo + k];
  return sum;
}

/**
 * Indices of the `k` largest values in `scores`, best first.
 *
 * Ties resolve by ascending index; an arbitrary choice at the k-th position changes
 * the neighbour set, and with it the clustering and the hubness metric.
 */
export function topK(scores, k, skipIndex = -1) {
  const idx = [];
  const keyed = new Float64Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    keyed[i] = Number.isFinite(scores[i]) ? rankKey(scores[i]) : scores[i];
    if (i !== skipIndex) idx.push(i);
  }
  idx.sort((a, b) => (keyed[b] - keyed[a]) || (a - b));
  return idx.slice(0, k);
}

/**
 * Match each query to its most similar reference by cosine.
 *
 * Scores one query row at a time; the full queries-by-references matrix exhausts a
 * browser tab's address space.
 *
 * @param {string[]} queries
 * @param {string[]} references
 * @param {object} queryEmb  matrix of query embeddings
 * @param {object} refEmb    matrix of reference embeddings
 * @param {number|null} threshold  below this a match is reported as none
 */
export function matchByEmbeddings(queries, references, queryEmb, refEmb, threshold = null) {
  const q = l2Normalize(queryEmb);
  const r = l2Normalize(refEmb);
  const out = [];
  for (let i = 0; i < q.rows; i++) {
    let best = -1;
    let bestScore = -Infinity;
    let bestKey = -Infinity;
    for (let j = 0; j < r.rows; j++) {
      const s = dot(q, i, r, j);
      // rounded so ties resolve the same on every runtime; strict > keeps the earlier,
      // as max.col(ties="first") and np.argmax do
      const keyed = rankKey(s);
      if (keyed > bestKey) {
        bestKey = keyed;
        bestScore = s;
        best = j;
      }
    }
    // quantized on both sides, as Python and R do
    const accepted = threshold === null || rankKey(bestScore) >= rankKey(threshold);
    out.push({
      given_entity: queries[i],
      alethia_prediction: accepted ? references[best] : null,
      alethia_score: accepted ? bestScore : null,
    });
  }
  return out;
}

/**
 * Merge edges between entities that are mutual nearest neighbours.
 *
 * Mutual agreement stops one hub from chaining every entity into a single cluster.
 */
export function mutualNnEdges(emb, { floor = 0.8, k = 5, requireMutual = true } = {}) {
  const x = l2Normalize(emb);
  const n = x.rows;
  if (n < 2) return [];
  const kk = Math.max(1, Math.min(k, n - 1));

  const neighbours = new Array(n);
  const cosines = new Array(n);
  const margin = new Float64Array(n);
  const scratch = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) scratch[j] = i === j ? -Infinity : dot(x, i, x, j);
    // one ordering serves both uses; the margin needs two even when k is 1
    const ranked = topK(scratch, Math.max(kk, 2), i);
    neighbours[i] = ranked.slice(0, kk);
    cosines[i] = neighbours[i].map((j) => scratch[j]);
    // Top-1 minus top-2: how decisively the best neighbour wins.
    margin[i] = scratch[ranked[0]] - scratch[ranked[1]];
  }

  const seen = new Set();
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < neighbours[i].length; c++) {
      const j = neighbours[i][c];
      const cosine = cosines[i][c];
      if (rankKey(cosine) < rankKey(floor)) continue;
      const mutual = neighbours[j].includes(i);
      if (requireMutual && !mutual) continue;
      const a = Math.min(i, j);
      const b = Math.max(i, j);
      const key = `${a}-${b}`;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push({
        i: a,
        j: b,
        cosine,
        margin_i: margin[a],
        margin_j: margin[b],
        mutual,
        confidence: cosine * (1 + Math.min(margin[a], margin[b])),
      });
    }
  }
  // total order; confidence alone leaves tied edges in construction order
  edges.sort((p, q2) => (q2.confidence - p.confidence) || (p.i - q2.i) || (p.j - q2.j));
  return edges;
}

/** Connected components of the edge graph, labelled in first-appearance order. */
export function connectedComponents(n, edges) {
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  for (const e of edges) {
    const ra = find(e.i);
    const rb = find(e.j);
    if (ra !== rb) parent[ra] = rb;
  }
  const roots = new Map();
  const labels = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!roots.has(r)) roots.set(r, roots.size);
    labels[i] = roots.get(r);
  }
  return labels;
}

/** Group entities into clusters, with a canonical name per cluster. */
export function clusterEntities(entities, emb, options = {}) {
  const unique = [...new Set(entities)];
  const edges = mutualNnEdges(emb, options);
  const filtered = options.minConfidence == null
    ? edges
    : edges.filter((e) => e.confidence >= options.minConfidence);
  const labels = connectedComponents(unique.length, filtered);

  const canonical = new Map();
  for (let i = 0; i < unique.length; i++) {
    const lab = labels[i];
    const current = canonical.get(lab);
    // Shortest member, usually the cleanest spelling.
    if (current === undefined || unique[i].length < current.length) {
      canonical.set(lab, unique[i]);
    }
  }
  return { entities: unique, labels, edges: filtered, canonical };
}

/** Population standard deviation, matching numpy's default ddof of 0. */
export function popStd(values) {
  const n = values.length;
  if (!n) return 0;
  let mu = 0;
  for (const v of values) mu += v;
  mu /= n;
  let acc = 0;
  for (const v of values) acc += (v - mu) ** 2;
  return Math.sqrt(acc / n);
}

export function mean(values) {
  if (!values.length) return NaN;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

export function median(values) {
  if (!values.length) return NaN;
  const sorted = Float64Array.from(values).sort();
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
