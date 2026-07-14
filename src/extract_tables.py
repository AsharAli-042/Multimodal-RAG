import os
import fitz
from PIL import Image


class TableExtractor:
    """
    Extract tables by cropping them as images instead of trying to
    parse their text.
    Later, Gemini Vision will convert these table images into
    clean markdown.
    """

    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.output_folder = "data/extracted/tables"
        os.makedirs(self.output_folder, exist_ok=True)
        self.metadata = []

    def extract(self):
        table_id = 1

        for page_number in range(len(self.doc)):
            page = self.doc.load_page(page_number)
            finder = page.find_tables()

            if finder is None:
                continue

            if len(finder.tables) == 0:
                continue

            for table in finder.tables:
                bbox = table.bbox
                pix = page.get_pixmap(
                    clip=bbox,
                    dpi=300
                )
                filename = os.path.join(
                    self.output_folder,
                    f"table_{table_id}.png"
                )

                pix.save(filename)
                self.metadata.append(
                    {
                        "table_id": table_id,
                        "page": page_number + 1,
                        "image": filename,
                        "type": "table"
                    }
                )

                print(f"Saved table {table_id} from page {page_number+1}")
                table_id += 1

        return self.metadata