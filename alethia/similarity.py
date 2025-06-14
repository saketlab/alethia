import faiss
import numpy as np


class FastSimilarityMatcher:
    def __init__(self, reference_embeddings, use_gpu=False):
        self.reference_embeddings = np.array(reference_embeddings).astype("float32")

        # Build FAISS index for fast similarity search
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.IndexFlatIP(self.reference_embeddings.shape[1])
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
        else:
            self.index = faiss.IndexFlatIP(self.reference_embeddings.shape[1])

        faiss.normalize_L2(self.reference_embeddings)
        self.index.add(self.reference_embeddings)

    def find_matches(self, query_embeddings, k=5):
        query_embeddings = np.array(query_embeddings).astype("float32")
        faiss.normalize_L2(query_embeddings)

        similarities, indices = self.index.search(query_embeddings, k)
        return similarities, indices
