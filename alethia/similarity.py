import numpy as np


class FastSimilarityMatcher:
    def __init__(self, reference_embeddings, use_gpu=False):
        import faiss

        self.reference_embeddings = np.array(reference_embeddings).astype("float32")
        dim = self.reference_embeddings.shape[1]

        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, faiss.IndexFlatIP(dim)
            )
        else:
            self.index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(self.reference_embeddings)
        self.index.add(self.reference_embeddings)

    def find_matches(self, query_embeddings, k=5):
        import faiss

        query_embeddings = np.array(query_embeddings).astype("float32")
        faiss.normalize_L2(query_embeddings)
        return self.index.search(query_embeddings, k)
