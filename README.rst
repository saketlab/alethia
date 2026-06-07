=======
alethia
=======
.. image:: docs/logo.png
   :alt: Logo
   :width: 200px
   :align: center

A Python package for entity matching, standardization, and visualization using embeddings from large language models.

Alethia cleans and standardizes messy entity data (location names, product names, medical
records) using semantic similarity and embedding visualizations.

Alethia is **model-agnostic**: pass any embedding model, either a model name string or a
plain ``embed_fn(list[str]) -> ndarray`` callable, and Alethia handles the matching. It can
also assess models without ground-truth labels, so you can check which one works best on
your own data.

Installation
============

You can install alethia via pip:

.. code-block:: bash

    pip install alethia

For development installation:


.. code-block:: bash

    conda create -n alethia python=3.12 pip
    conda activate alethia
    git clone https://github.com/saketlab/alethia.git
    cd alethia
    pip install -e .



Quick Start
===========

Basic Entity Matching
---------------------

.. code-block:: python

    import pandas as pd
    from alethia import alethia

    # Load your data
    df = pd.read_csv("your_data.csv")

    # Define reference entries (correct, standardized entities)
    reference_entries = [x for x in list(set(df["correct_column"])) if str(x) != "nan"]

    # Match incorrect entries against reference entries
    incorrect_entries = df["incorrect_column"].tolist()
    alethia_output = alethia(incorrect_entries, reference_entries)

    # View entries that were corrected
    corrected = alethia_output[alethia_output.given_entity != alethia_output.alethia_prediction]
    print(corrected)

Bring Your Own Model
--------------------

Pass any model as a name string or a callable that returns embeddings:

.. code-block:: python

    import numpy as np
    from alethia import alethia

    # (a) by name (resolved via sentence-transformers / fastembed if installed)
    alethia(incorrect_entries, reference_entries, model="all-MiniLM-L6-v2")

    # (b) by callable: any embedding source, no heavy dependencies
    def embed_fn(texts: list[str]) -> np.ndarray:
        ...  # return shape (len(texts), dim)
    alethia(incorrect_entries, reference_entries, model=embed_fn)

    # (c) by a loaded model object (SentenceTransformer or any HuggingFace sentence model)
    from sentence_transformers import SentenceTransformer
    alethia(incorrect_entries, reference_entries, model=SentenceTransformer("all-MiniLM-L6-v2"))

Choosing a model (label-free assessment)
----------------------------------------

You can measure which embedding model fits your data instead of guessing, with no labels:

.. code-block:: python

    from alethia import assess_models

    report = assess_models(
        queries=incorrect_entries,
        references=reference_entries,
        models={
            "minilm": "all-MiniLM-L6-v2",
            "mpnet": "all-mpnet-base-v2",
        },
    )

    print(report.to_table())          # tidy, score-sorted DataFrame
    print("Best:", report.best.name)  # recommended model for YOUR data
    report.to_html("assessment.html") # self-contained HTML report with charts

Visualizing Entity Embeddings
-----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from alethia import (
        do_pca,
        do_umap,
        get_embeddings,
        load_sentence_transformer,
        plot_embedding
    )

    # Load a sentence transformer model
    model = load_sentence_transformer("Salesforce/SFR-Embedding-Mistral")

    # Create embeddings for your entities
    entities = ["Entity 1", "Entity 2", "Entity 3", ...]
    embeddings = get_embeddings(texts=entities, model=model, show_progress=True)

    # Dimensionality reduction with PCA
    pca, exp_var = do_pca(embeddings, return_expl_var=True)

    # Dimensionality reduction with UMAP
    umap = do_umap(embeddings)

    # Plot the results
    plot_embedding(
        pca,
        labels=entities,
        dims=[1, 2],
        title="PCA of Entity Embeddings",
        explained_var=exp_var
    )

    plot_embedding(
        umap,
        labels=entities,
        dims=[1, 2],
        title="UMAP of Entity Embeddings"
    )

Features
========

Entity Matching and Standardization
-----------------------------------

* Align messy entity names with a reference list of standardized entries
* Based on semantic similarity using transformer embeddings
* Handles typos, abbreviations, and other common data entry inconsistencies

Embedding Analysis
------------------

* Generate embeddings for text entities using state-of-the-art models
* Reduce dimensionality with PCA or UMAP for visualization and analysis
* Identify clusters and outliers in your entity data

Visualization
-------------

* Plot embeddings with customizable visualizations
* Compare different embedding projections

Use Cases
=========

* Standardizing location names across disparate datasets
* Aligning entity records from multiple sources
* Exploring semantic relationships between entities

License
=======

MIT

Contributing
============

Contributions are welcome! Please feel free to submit a pull request.
