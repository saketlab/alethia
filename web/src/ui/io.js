/** Split pasted or uploaded text into entries, one per line. */
export function parseList(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    // single-column CSV exports often keep their header and quotes
    .map((line) => line.replace(/^"(.*)"$/, '$1'))
    .filter((line) => line.length > 0 && line.toLowerCase() !== 'nan');
}

/** Save results as CSV, quoting properly so entries containing commas survive. */
export function downloadCsv(rows, filename) {
  const escape = (v) => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = 'given_entity,alethia_prediction,alethia_score';
  const body = rows
    .map((r) => [r.given_entity, r.alethia_prediction, r.alethia_score].map(escape).join(','))
    .join('\n');
  const blob = new Blob([`${header}\n${body}\n`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
