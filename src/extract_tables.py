import os
import json
import fitz


class TableExtractor:
    """
    Extract tables from the PDF.

    PyMuPDF's table extraction is simple but works well
    for research papers.
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
            try:
                finder = page.find_tables()

            except Exception:
                continue

            if finder is None:
                continue

            for table in finder.tables:
                rows = table.extract()
                filename = os.path.join(
                    self.output_folder,
                    f"table_{table_id}.json"
                )

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(
                        rows,
                        f,
                        indent=4
                    )

                self.metadata.append(
                    {
                        "table_id": table_id,
                        "page": page_number + 1,
                        "file": filename,
                        "type": "table"
                    }
                )

                table_id += 1

        with open("data/extracted/tables_metadata.json","w",encoding="utf-8") as f:

            json.dump(
                self.metadata,
                f,
                indent=4
            )

        print(f"Extracted {len(self.metadata)} tables.")