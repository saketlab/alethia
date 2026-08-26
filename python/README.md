# alethia

<p align="center">
  <img src="https://raw.githubusercontent.com/saketlab/alethia/main/docs/assets/logo.png" alt="alethia" width="200">
</p>

[![PyPI](https://img.shields.io/pypi/v/alethia.svg)](https://pypi.org/project/alethia/)
[![Python](https://img.shields.io/pypi/pyversions/alethia.svg)](https://pypi.org/project/alethia/)
[![License](https://img.shields.io/pypi/l/alethia.svg)](https://github.com/saketlab/alethia/blob/main/LICENSE)
[![Docs](https://github.com/saketlab/alethia/actions/workflows/deploy.yml/badge.svg)](https://alethia.saketlab.org/cli/python/)

Match messy entity names against a reference list.


Documentation: <https://alethia.saketlab.org/cli/python/>  |  Try it in your browser: <https://alethia.saketlab.org/>

## Install

```bash
pip install alethia
```

## Command line


```bash
# Fix messy names against a reference list
alethia match messy.csv reference.csv

# Match by meaning rather than spelling, and save the result
alethia match messy.csv reference.csv --model all-MiniLM-L6-v2 -o fixed.csv

# Group duplicates when you have no reference list
alethia cluster companies.csv

# Work out which embedding model does best on your data
alethia assess messy.csv reference.csv -m all-MiniLM-L6-v2 -m BAAI/bge-small-en-v1.5
```

Files can be `.csv`, `.tsv`, `.xlsx`, or `.txt` (one entity per line). alethia tries to infer the
text column.

## Python

### Matching

```python
from alethia import alethia

result = alethia(
    ["New Yrok", "Los Angelos", "Chicagoo"],
    ["New York", "Los Angeles", "Chicago"],
)
print(result)
#   given_entity alethia_prediction  alethia_score alethia_method alethia_backend
# 0     New Yrok           New York          0.875      rapidfuzz       rapidfuzz
# 1  Los Angelos        Los Angeles          0.909      rapidfuzz       rapidfuzz
# 2     Chicagoo            Chicago          0.933      rapidfuzz       rapidfuzz
```

Pass `model=` to switch backends: `"rapidfuzz"` (default, string similarity),
any embedding model name, `"openai"`, `"gemini"`, a loaded model object, or your own
`embed_fn(list[str]) -> ndarray`.

### Label-free model assessment

Which embedding model should you use? 

```python
from alethia.assess import assess_models

report = assess_models(
    queries=messy_names,
    references=canonical_names,
    models={
        "minilm": "all-MiniLM-L6-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
    },
)
print(report.best.name)
report.to_html("assessment.html")
```

The composite score is based on geometric probes of the embedding space which includes separability,
alignment under realistic noise, retrieval margin, hubness, mutual-nearest-neighbour
rate weighted within family. The score is comparative, so it needs at least two models.

### Clustering

```python
from alethia import cluster_entities

result = cluster_entities(messy_names, model="all-MiniLM-L6-v2")
for canonical, members in result.clusters().items():
    print(canonical, members)
```

Edges require mutual nearest neighbourhood above a cosine score and carry a
confidence
