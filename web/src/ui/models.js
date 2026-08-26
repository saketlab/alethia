// Models with ONNX weights transformers.js can load; a subset of the Python roster.
// Sizes are the int8-quantised download, so browser scores are not bit-identical to
// the server packages. pooling is a training property, not a knob; omitted means mean.
// BioLORD-2023-C is excluded on licence: UMLS terms, no published ONNX, so converting
// it would redistribute SNOMED-derived weights. Numbers in alethia-datasets.
/**
 * Minimum cosine to accept a match, unless the user overrides it.
 *
 * Matches DEFAULT_THRESHOLD in the Python and R packages. The right cutoff is
 * corpus-specific; clearing the field disables it.
 */
export const DEFAULT_THRESHOLD = 0.7;

/**
 * The model a run uses when the URL names none.
 *
 * Distinct from the `default` flag below, which marks the compare list's pre-ticked
 * models.
 */
export const DEFAULT_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

/**
 * Which model an id refers to, falling back to the default.
 *
 * The shell warms a model on load and the match view runs one; if the two disagree the
 * warm-up downloads a model nothing uses.
 */
export function resolveModel(id) {
  return MODELS.find((m) => m.id === id)
    ?? MODELS.find((m) => m.id === DEFAULT_MODEL_ID);
}

// default: true pre-ticks in the compare list; the match fallback is DEFAULT_MODEL_ID
export const MODELS = [
  {
    id: 'Xenova/all-MiniLM-L6-v2',
    label: 'MiniLM-L6',
    size: '23 MB',
    note: 'Fast and small. A sensible default for short names.',
    default: true,
  },
  {
    // 57.0% mean top-1 on ICD-10 against GTE-small's 61.2%, at three times the download
    id: 'Xenova/all-mpnet-base-v2',
    label: 'mpnet-base',
    size: '109 MB',
    note: 'A larger general model from a different family. Useful as a second opinion.',
    default: false,
  },
  {
    id: 'Xenova/bge-small-en-v1.5',
    label: 'BGE-small',
    size: '34 MB',
    note: 'Strong retrieval model at a small size.',
    default: false,
  },
  {
    // 61.2% mean top-1 on ICD-10, beating mpnet-base at under a third of the download
    id: 'Xenova/gte-small',
    label: 'GTE-small',
    size: '34 MB',
    note: 'Small and the most accurate of the light models here on clinical text.',
    default: true,
  },
  {
    id: 'Xenova/multilingual-e5-small',
    label: 'mE5-small',
    size: '118 MB',
    note: 'Multilingual. Pick this if your entries are not all English.',
    default: false,
  },
  {
    // best browser-capable model at 62.9% mean top-1, but ten times GTE-small's download
    id: 'Xenova/gte-large',
    label: 'GTE-large',
    size: '336 MB',
    note: 'The most accurate option here on descriptive clinical text, but ten times '
      + 'the download of GTE-small for a small gain. Worth it only if accuracy matters '
      + 'more than the wait.',
    default: false,
  },
  {
    // biomedical specialist; 110 MB against 57 MB for both defaults together
    id: 'Xenova/SapBERT-from-PubMedBERT-fulltext',
    label: 'SapBERT',
    size: '110 MB',
    pooling: 'cls',
    note: 'Biomedical. Trained on UMLS to align names for the same concept, so it '
      + 'connects brand drugs and chemical names the general models miss. Weaker on '
      + 'descriptive clinical phrases than on entity names.',
    default: false,
  },
];
