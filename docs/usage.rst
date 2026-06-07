=====
Usage
=====

Basic matching
--------------

.. code-block:: python

   from alethia import alethia

   references = ["New York", "Los Angeles", "Chicago", "Houston"]
   dirty = ["New Yrok", "Los Anglees", "Chicagoo"]

   result = alethia(dirty, references, model="all-MiniLM-L6-v2")
   print(result[["given_entity", "alethia_prediction", "alethia_score"]])

``result`` is a :class:`pandas.DataFrame` with one row per query, plus run metadata in
``result.attrs`` (backend, processing time, exact-match counts).

Bring your own model
--------------------

Any callable that maps a list of strings to an ``(n, d)`` array works as a model, with no
heavy dependencies required:

.. code-block:: python

   import numpy as np

   def embed_fn(texts: list[str]) -> np.ndarray:
       ...  # your embeddings, shape (len(texts), dim)

   alethia(dirty, references, model=embed_fn)

Choosing a model, label-free
----------------------------

.. code-block:: python

   from alethia import assess_models

   report = assess_models(
       queries=dirty,
       references=references,
       models={"minilm": "all-MiniLM-L6-v2", "mpnet": "all-mpnet-base-v2"},
   )

   report.to_table()             # tidy, score-sorted DataFrame
   report.best.name              # recommended model for your data
   report.to_html("report.html") # self-contained HTML report

Only ``references`` are required; pass ``queries=[]`` to assess models on the reference
structure alone. ``report.to_table()`` lists every metric the score is built from.
