import os
import json
import pickle
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEYS = []

i = 1
while True:
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key is None:
        break
    API_KEYS.append(key)
    i += 1

current_key = 0
client = genai.Client(api_key=API_KEYS[current_key])

MODEL = "gemini-embedding-001"
INPUT = "data/extracted/documents.json"
OUTPUT = "output/metadata.pkl"

class Embedder:
    def __init__(self):
        with open(INPUT, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def embed(self):
        vectors = []
        metadata = []
        start_index = 0

        os.makedirs("output", exist_ok=True)

        if os.path.exists(OUTPUT):
            with open(OUTPUT, "rb") as f:
                data = pickle.load(f)
                vectors = data["vectors"]
                metadata = data["metadata"]
                start_index = len(vectors)
                print(f"Resuming from document {start_index + 1}")

        print("=" * 60)
        print("Creating Embeddings")
        print("=" * 60)

        for i in range(start_index, len(self.documents)):
            doc = self.documents[i]
            result = client.models.embed_content(
                model=MODEL,
                contents=doc["content"]
            )

            vector = result.embeddings[0].values
            vectors.append(vector)
            metadata.append(doc)
            with open(OUTPUT, "wb") as f:
                pickle.dump(
                    {
                        "vectors": vectors,
                        "metadata": metadata
                    },
                    f
                )

            print(f"{i+1}/{len(self.documents)} Saved")

        os.makedirs("output", exist_ok=True)

        with open(OUTPUT, "wb") as f:
            pickle.dump(
                {
                    "vectors": vectors,
                    "metadata": metadata
                },
                f
            )
        print("\nEmbeddings Saved.")

if __name__ == "__main__":
    Embedder().embed()