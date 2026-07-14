# Multimodal RAG on *Attention Is All You Need*

A Multimodal Retrieval-Augmented Generation (RAG) system built using Google's Gemini API. This project extracts textual and visual information from the **Attention Is All You Need** research paper, embeds all extracted content into a shared vector space, retrieves the most relevant information for a user query, and generates grounded answers using Gemini.

The implementation demonstrates retrieval across multiple modalities, including text, headings, tables, and figures.

---

# Features

- Extracts body text from the research paper
- Detects and stores document headings
- Extracts figures and diagrams from the PDF
- Extracts tables from the PDF
- Uses Gemini Vision to understand figures and tables
- Generates embeddings for all extracted content
- Stores embeddings inside a FAISS vector database
- Retrieves the most relevant information for a query
- Produces grounded answers using Gemini
- Logs every query, retrieved context, and generated answer

---

# Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Internet connection (required for Gemini API)

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
cd Multimodal-RAG
```

Create Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

Do **not** commit your `.env` file to GitHub.

---

# Dataset

Download the original paper:

**Attention Is All You Need**

Place it inside the `data` directory.

---

# Project Structure

```
┣ 📂data
 ┃ ┣ 📂extracted
 ┃ ┗ 📜Attention_is_all_you_need.pdf
 ┣ 📂output
 ┣ 📂src
 ┃ ┣ 📜chunker.py
 ┃ ┣ 📜embed.py
 ┃ ┣ 📜extract.py
 ┃ ┣ 📜extract_tables.py
 ┃ ┣ 📜rag.py
 ┃ ┣ 📜vector_store.py
 ┃ ┗ 📜vision.py
 ┣ 📜.env
 ┣ 📜.gitignore
 ┣ 📜app.py
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

---

# Running the Project

Run the complete pipeline using:

```bash
python app.py
```

The pipeline performs the following steps automatically:

1. Extract text, headings, and images from the PDF
2. Extract tables
3. Generate descriptions for figures and tables using Gemini Vision
4. Generate embeddings for all extracted content
5. Build the FAISS vector database
6. Execute sample queries through the RAG pipeline

---

# Output

The project generates:

- Extracted text files
- Extracted figures
- Extracted tables
- Structured metadata
- Vision-generated descriptions
- Embedding vectors
- FAISS vector index
- Query logs containing retrieved context and generated answers
