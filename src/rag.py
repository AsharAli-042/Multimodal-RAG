import pickle
import faiss
import numpy as np
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY_1")
)

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.1-flash-lite"

INDEX_PATH = "output/index.faiss"
METADATA_PATH = "output/metadata.pkl"

TOP_K = 5

class MultimodalRAG:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)

        self.metadata = data["metadata"]

    def embed_query(self, question):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=question
        )

        return np.array(
            result.embeddings[0].values,
            dtype="float32"
        ).reshape(1, -1)

    def retrieve(self, question):
        query_vector = self.embed_query(question)
        distances, indices = self.index.search(
            query_vector,
            TOP_K
        )

        retrieved = []
        for idx in indices[0]:
            retrieved.append(self.metadata[idx])

        return retrieved

    def build_context(self, docs):
        context = ""
        for i, doc in enumerate(docs, start=1):
            context += f"""
============ DOCUMENT {i} ============

TYPE:
{doc["type"]}

SOURCE:
{doc["file"]}

CONTENT:
{doc["content"]}

"""
        return context

    def ask(self, question):
        docs = self.retrieve(question)
        context = self.build_context(docs)

        prompt = f"""
You are answering questions ONLY using the provided context.

If the answer is not contained in the context,
say:

"I could not find that information in the paper."

Context:

{context}

Question:

{question}
                """

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )

        print("=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        print()

        print("=" * 80)
        print("RETRIEVED CONTEXT")
        print("=" * 80)
        print(context)

        print()

        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(response.text)

        print("\n")

if __name__ == "__main__":
    rag = MultimodalRAG()
    rag.ask("What is Multi-Head Attention?")
    rag.ask("Explain the Transformer encoder-decoder architecture shown in the figure.")
    rag.ask("What BLEU score did the Transformer Base model achieve?")