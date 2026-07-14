import os
import json
import fitz  # PyMuPDF
from PIL import Image

class PDFExtractor:
    """
    Extracts multimodal content from a research paper.

    Output:
        data/extracted/
            text/
            images/
            metadata.json
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

        self.output_root = "data/extracted"
        self.text_folder = os.path.join(self.output_root, "text")
        self.image_folder = os.path.join(self.output_root, "images")

        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.text_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)

        self.doc = fitz.open(pdf_path)

        self.metadata = {
            "pages": len(self.doc),
            "text_chunks": [],
            "images": [],
            "headings": []
        }

    """ TEXT EXTRACTION """  
    
    def extract_text(self):
        print("=" * 60)
        print("Extracting Text")
        print("=" * 60)

        for page_number, page in enumerate(self.doc):
            text = page.get_text("text")

            filename = os.path.join(
                self.text_folder,
                f"page_{page_number + 1}.txt"
            )

            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

            self.metadata["text_chunks"].append(
                {
                    "page": page_number + 1,
                    "file": filename
                }
            )
            print(f"Saved Page {page_number+1}")

    """ HEADING EXTRACTION """
    
    def extract_headings(self):
        print("=" * 60)
        print("Finding Headings")
        print("=" * 60)

        for page_number, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = ""
                    max_size = 0

                    for span in line["spans"]:
                        text += span["text"]

                        if span["size"] > max_size:
                            max_size = span["size"]

                    text = text.strip()
                    if len(text) == 0:
                        continue

                    # Very simple heuristic:
                    # Larger fonts are probably headings.
                    if max_size >= 14:
                        self.metadata["headings"].append(
                            {
                                "page": page_number + 1,
                                "text": text,
                                "font_size": round(max_size, 2)
                            }
                        )
        print(
            f"Found {len(self.metadata['headings'])} possible headings."
        )

    """ IMAGE EXTRACTION """
    
    def extract_images(self):
        print("=" * 60)
        print("Extracting Images")
        print("=" * 60)

        image_counter = 1

        for page_number in range(len(self.doc)):
            page = self.doc.load_page(page_number)
            image_list = page.get_images(full=True)

            print(
                f"Page {page_number+1}: {len(image_list)} images"
            )

            for img in image_list:
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                extension = base_image["ext"]

                filename = os.path.join(
                    self.image_folder,
                    f"figure_{image_counter}.{extension}"
                )

                with open(filename, "wb") as f:
                    f.write(image_bytes)

                self.metadata["images"].append(
                    {
                        "page": page_number + 1,
                        "file": filename
                    }
                )
                image_counter += 1

        print(
            f"Extracted {len(self.metadata['images'])} images."
        )

    """ SAVE METADATA """
    
    def save_metadata(self):
        metadata_path = os.path.join(
            self.output_root,
            "metadata.json"
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                self.metadata,
                f,
                indent=4
            )

        print("Metadata saved.")

    """ PIPELINE """
    def run(self):
        self.extract_text()
        self.extract_headings()
        self.extract_images()
        self.save_metadata()

        print("=" * 60)
        print("Extraction Complete")
        print("=" * 60)


if __name__ == "__main__":
    extractor = PDFExtractor(
        "data/Attention_Is_All_You_Need.pdf"
    )
    extractor.run()