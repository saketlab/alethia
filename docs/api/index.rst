API Reference
=============

Core matching
-------------

.. autofunction:: alethia.alethia

.. autofunction:: alethia.get_best_available_backend

.. autofunction:: alethia.check_optional_dependencies

Model-agnostic embedding interface
----------------------------------

.. autoclass:: alethia.Embedder
   :members:

.. autoclass:: alethia.CallableEmbedder
   :members:

.. autofunction:: alethia.as_embedder

.. autofunction:: alethia.match_by_embeddings

Label-free model assessment
---------------------------

.. autofunction:: alethia.assess_models

.. autofunction:: alethia.assessment_table

.. autoclass:: alethia.AssessmentReport
   :members:

.. autoclass:: alethia.ModelAssessment
   :members:

Embeddings & visualization
--------------------------

.. autofunction:: alethia.get_embeddings

.. autofunction:: alethia.do_pca

.. autofunction:: alethia.do_umap

.. autofunction:: alethia.plot_embedding

.. autofunction:: alethia.plot_embedding_df
