# Multimodal RAG on *Attention Is All You Need*

A **Multimodal Retrieval-Augmented Generation (RAG)** system built with **Google Gemini** that performs grounded question answering over the **Attention Is All You Need** research paper.

The pipeline extracts textual and visual information from the paper, understands figures and tables using Gemini Vision, embeds every modality into a shared vector space, retrieves the most relevant context for a query, and generates answers grounded only in the retrieved information.

---

## Features

- Extracts body text from the research paper
- Detects document headings
- Extracts figures and diagrams
- Extracts tables from the PDF
- Uses Gemini Vision to understand visual content
- Generates embeddings for text, headings, figures, and tables
- Stores embeddings in a FAISS vector database
- Retrieves relevant context using semantic similarity search
- Generates grounded answers with Gemini
- Logs every query, retrieved context, and generated answer

---

## Architecture

```text
                  Attention_Is_All_You_Need.pdf
                               │
                               ▼
                         extract.py
                               │
          ┌────────────┬──────────────┬─────────────┐
          ▼            ▼              ▼             ▼
     Text Chunks    Headings       Figures       Tables
          │            │              │             │
          └──────┬─────┴──────────────┴─────────────┘
                 ▼
             vision.py
      (Describe Figures & Tables)
                 │
                 ▼
              embed.py
                 │
                 ▼
          vector_store.py
             (FAISS Index)
                 │
                 ▼
               rag.py
      Query → Retrieve → Gemini → Answer
```

The project follows a complete **Multimodal RAG pipeline**. The research paper is first processed to extract textual and visual content. Figures and tables are interpreted using Gemini Vision, while text is chunked for semantic retrieval. All extracted information is embedded into a shared vector space and indexed using FAISS. During inference, the most relevant context is retrieved and supplied to Gemini to generate grounded responses.

---

## Project Structure

```text
┣ 📂data
┃ ┣ 📂extracted
┃ ┃ ┣ 📂chunks
┃ ┃ ┣ 📂images
┃ ┃ ┣ 📂tables
┃ ┃ ┗ 📂text
┃ ┗ 📜Attention_Is_All_You_Need.pdf
┣ 📂output
┃ ┣ 📂index
┃ ┣ 📂results
┃ ┣ 📜index.faiss
┃ ┗ 📜metadata.pkl
┣ 📂src
┃ ┣ 📜chunker.py
┃ ┣ 📜embed.py
┃ ┣ 📜extract.py
┃ ┣ 📜extract_tables.py
┃ ┣ 📜rag.py
┃ ┣ 📜vector_store.py
┃ ┗ 📜vision.py
┣ 📜app.py
┣ 📜requirements.txt
┣ 📜README.md
┣ 📜.gitignore
┗ 📜.env
```

---

## Source Files

| File | Purpose |
|------|---------|
| **extract.py** | Extracts body text, headings, figures, and metadata from the research paper. |
| **chunker.py** | Splits extracted text into overlapping semantic chunks suitable for embedding. |
| **extract_tables.py** | Detects and extracts tables from the PDF as images. |
| **vision.py** | Uses Gemini Vision to generate descriptions for figures and tables. |
| **embed.py** | Generates Gemini embeddings for every extracted document and stores them for indexing. |
| **vector_store.py** | Builds and saves the FAISS vector index used for semantic retrieval. |
| **rag.py** | Retrieves relevant context from FAISS and generates grounded answers using Gemini. |
| **app.py** | Runs the complete pipeline with a single command. |

---

## Installation

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

## Dataset

Download the original research paper:

**Attention Is All You Need**

Rename the downloaded file as:

```text
Attention_Is_All_You_Need.pdf
```

Place the PDF inside the **data/** directory.

---

## Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

Never commit your API key or `.env` file to version control.

---

## Running the Project

Execute the entire pipeline using:

```bash
python app.py
```

The wrapper script performs the following steps automatically:

1. Extract text, headings, figures, and metadata
2. Extract tables from the PDF
3. Generate Gemini Vision descriptions for figures and tables
4. Generate embeddings for every extracted document
5. Build the FAISS vector index
6. Execute the Multimodal RAG pipeline

---

## Output

Running the project generates:

- Extracted page text
- Semantic text chunks
- Extracted figures
- Extracted tables
- Vision-generated figure descriptions
- Vision-generated table descriptions
- Document metadata
- Embedding vectors
- FAISS vector index
- Query logs containing:
  - User query
  - Retrieved context
  - Generated answer

---

## Example Workflow

```text
Research Paper
      │
      ▼
Extraction
      │
      ▼
Vision Understanding
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Vector Store
      │
      ▼
User Query
      │
      ▼
Semantic Retrieval
      │
      ▼
Gemini
      │
      ▼
Grounded Answer
```

---

## Future Improvements

- Hybrid retrieval (semantic + keyword search)
- Better table parsing with structured extraction
- Cross-modal retrieval ranking
- Web interface using Streamlit or Gradio
- Support for multiple research papers
- Metadata-aware filtering during retrieval