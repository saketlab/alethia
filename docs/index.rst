=============================
Alethia Documentation
=============================

.. image:: logo.svg
   :alt: Alethia Logo
   :width: 200px
   :align: center

|

**Alethia** is a Python package for entity matching, standardization, and visualization using embeddings from large language models.

Alethia helps you clean and standardize messy entity data (like location names, product names, medical records, etc.) by leveraging semantic similarity and embedding visualizations.

Key Features
============

**Entity Matching & Standardization**
  Align messy entity names with a reference list using semantic similarity

**Multiple Embedding Backends**
  Support for sentence-transformers, FastEmbed, OpenAI, and Google Gemini

**Fuzzy Matching**
  Built-in fuzzy matching with rapidfuzz and LLM-based alternatives

**Visualization**
  Reduce dimensionality with PCA/UMAP and create interactive plots

**Model Recommendations**
  Intelligent model selection based on MTEB benchmark data

Quick Example
=============

.. code-block:: python

    import pandas as pd
    from alethia import alethia

    # Load your data
    df = pd.read_csv("your_data.csv")

    # Define reference entries (correct, standardized entities)
    reference_entries = [x for x in set(df["correct_column"]) if str(x) != "nan"]

    # Match incorrect entries against reference entries
    incorrect_entries = df["incorrect_column"].tolist()
    result = alethia(incorrect_entries, reference_entries)

    # View corrected entries
    corrected = result[result.given_entity != result.alethia_prediction]
    print(corrected)

Installation
============

Install the base package:

.. code-block:: bash

    pip install alethia

Or install with specific backends:

.. code-block:: bash

    # CPU-optimized installation
    pip install alethia[cpu]

    # GPU support
    pip install alethia[gpu]

    # Full installation (all features)
    pip install alethia[full]

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/entity_matching
   guide/embeddings
   guide/fuzzy_matching
   guide/visualization
   guide/model_selection

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/alethia
   api/embeddings
   api/similarity
   api/fuzzy_match
   api/stats
   api/models
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_matching
   examples/medical_coding
   examples/location_standardization

.. toctree::
   :maxdepth: 1
   :caption: Reference

   history
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
