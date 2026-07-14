import os
import json
from PIL import Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.1-flash-lite"

class VisionProcessor:
    def __init__(self):
        self.image_folder = "data/extracted/images"
        self.table_folder = "data/extracted/tables"

        self.documents = []

    def describe_figures(self):
        print("=" * 60)
        print("Understanding Figures")
        print("=" * 60)

        if not os.path.exists(self.image_folder):
            return

        images = sorted(os.listdir(self.image_folder))

        for image_name in images:
            path = os.path.join(
                self.image_folder,
                image_name
            )
            image = Image.open(path)

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    image,
                    """
                    You are analyzing a figure from the research paper
                    'Attention Is All You Need'.

                    Describe:

                    - what the figure illustrates
                    - important components
                    - labels
                    - arrows
                    - relationships

                    Keep the description under 200 words.
                    """
                ]
            )

            description = response.text.strip()
            print(f"{image_name} processed.")
            self.documents.append(
                {
                    "type": "figure",
                    "file": path,
                    "content": description
                }
            )

    def describe_tables(self):
        print("=" * 60)
        print("Understanding Tables")
        print("=" * 60)

        if not os.path.exists(self.table_folder):
            return

        tables = sorted(os.listdir(self.table_folder))

        for table_name in tables:
            path = os.path.join(
                self.table_folder,
                table_name
            )
            image = Image.open(path)

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    image,
                    """
                    This image contains a table from the research paper
                    'Attention Is All You Need'.

                    If it is readable:

                    1. Convert it into Markdown.
                    2. Explain what the table compares.
                    3. Mention the important numerical results.

                    If the image is unreadable, simply reply:

                    UNREADABLE
                    """
                ]
            )

            if response.text:
                markdown = response.text.strip()
            else:
                markdown = "Gemini could not extract this table."

            print(f"{table_name} processed.")
            self.documents.append(
                {
                    "type": "table",
                    "file": path,
                    "content": markdown
                }
            )

    def add_text_chunks(self):
        print("=" * 60)
        print("Loading Text Chunks")
        print("=" * 60)

        chunk_folder = "data/extracted/chunks"
        files = sorted(os.listdir(chunk_folder))

        for file in files:
            path = os.path.join(
                chunk_folder,
                file
            )

            with open(path,"r",encoding="utf-8") as f:
                text = f.read()

            self.documents.append(
                {
                    "type": "text",
                    "file": path,
                    "content": text
                }
            )
        print(f"Loaded {len(files)} chunks.")

    def save(self):
        output = "data/extracted/documents.json"

        with open(output,"w",encoding="utf-8") as f:
            json.dump(
                self.documents,
                f,
                indent=4,
                ensure_ascii=False
            )
        print(f"\nSaved {len(self.documents)} documents.")

    def run(self):
        self.add_text_chunks()
        self.describe_figures()
        self.describe_tables()
        self.save()


if __name__ == "__main__":
    processor = VisionProcessor()
    processor.run()