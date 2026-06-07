alethia: model-agnostic entity matching with language-model embeddings
======================================================================

.. image:: logo.svg
   :alt: alethia logo
   :width: 180px
   :align: center

|

**alethia** maps messy entity names (locations, drugs, medical records) to a canonical list
using embeddings. Bring any model, by name or as a plain function, and it handles the rest.

It can also tell you which model fits your data best, without any labels.

Install
-------

.. code-block:: bash

   pip install alethia

Match
-----

.. code-block:: python

   from alethia import alethia

   references = ["New York", "Los Angeles", "Chicago"]
   dirty = ["New Yrok", "Los Anglees", "Chicagoo"]

   alethia(dirty, references, model="all-MiniLM-L6-v2")

Pick a model for your data
--------------------------

.. code-block:: python

   from alethia import assess_models

   report = assess_models(dirty, references, models={
       "minilm": "all-MiniLM-L6-v2",
       "mpnet": "all-mpnet-base-v2",
   })
   report.best.name          # recommended model, chosen without labels
   report.to_html("report.html")

The examples below work through real public-health datasets.

.. toctree::
   :maxdepth: 1
   :caption: Get started

   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/01_india_district_harmonization
   notebooks/02_medication_name_standardization
   notebooks/03_validating_the_assessor

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/index
   contributing
   history
