import pickle
import faiss
import numpy as np
import os

INPUT = "output/metadata.pkl"
INDEX = "output/index.faiss"

class VectorStore:
    def build(self):
        with open(INPUT, "rb") as f:
            data = pickle.load(f)

        vectors = np.array(
            data["vectors"],
            dtype="float32"
        )

        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        faiss.write_index(index, INDEX)

        print("=" * 60)
        print("Vector Store Built")
        print("=" * 60)

        print(index.ntotal)

if __name__ == "__main__":
    VectorStore().build()