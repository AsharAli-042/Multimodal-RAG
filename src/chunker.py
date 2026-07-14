import os
import json

class TextChunker:
    """
    Splits extracted page text into smaller chunks suitable
    for embedding.
    """

    def __init__(
        self,
        text_folder="data/extracted/text",
        output_folder="data/extracted/chunks",
        chunk_size=800,
        overlap=150,
    ):

        self.text_folder = text_folder
        self.output_folder = output_folder

        self.chunk_size = chunk_size
        self.overlap = overlap

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(text_folder, exist_ok=True)
        self.chunk_metadata = []

    def chunk_text(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks

    def process(self):
        files = sorted(os.listdir(self.text_folder))
        chunk_id = 1

        for filename in files:
            path = os.path.join(self.text_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.chunk_text(text)
            page = int(filename.split("_")[1].split(".")[0])

            for chunk in chunks:
                out_file = os.path.join(
                    self.output_folder,
                    f"chunk_{chunk_id}.txt"
                )

                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(chunk)

                self.chunk_metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "file": out_file,
                        "type": "text"
                    }
                )

                chunk_id += 1

        with open("data/extracted/chunks_metadata.json","w",encoding="utf-8") as f:
            json.dump(
                self.chunk_metadata,
                f,
                indent=4
            )

        print(f"Created {len(self.chunk_metadata)} chunks.")