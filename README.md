---
title: Shromik QA - Bangladesh Labour Act RAG
emoji: ⚡
colorFrom: green
colorTo: black
sdk: docker
app_port: 8000
pinned: false
---

# Bangladesh Labour Act 2006 RAG Chatbot


An authoritative, grounded Retrieval-Augmented Generation (RAG) chatbot for the **Bangladesh Labour Act 2006** (including its 2013 and 2018 amendments, and Labour Rules 2015).

Built following a deliberately right-sized, single linear pipeline architecture — in-process **FAISS** vector search combined with an explicit **Section-Number Regex Override** to guarantee traceable legal citations and eliminate hallucinations.

---

## 🛠️ Tech Stack & Architecture

- **Corpus**: Bangladesh Labour Act 2006 (Consolidated up to 2018 amendments & Labour Rules 2015).
- **Text Extraction & Cleaning**: `pdfplumber` + Regex text cleaner stripping page headers, footers, gazette notices, and numbering artifacts.
- **Section Chunking**: Statutory boundary parsing (`Section [X]. [Title]`). Each chunk corresponds to a single citable legal unit.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense embeddings).
- **Vector Store**: `FAISS` (`faiss.IndexFlatIP` with L2-normalized vectors for cosine similarity) in-process without separate DB server.
- **RAG Orchestration**: Dense FAISS search + direct Section-Number Regex Override (`Section 27`, `Sec 100`, etc.).
- **Generation & LLM**: Anthropic API (`claude-3-5-sonnet-20240620`), Google Gemini (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`), or Groq (`llama-3.3-70b-versatile`).
- **Chatbot UI**: Streamlit application with interactive grounding verification expanders.
- **Evaluation**: Hand-labelled benchmark test set computing **Recall@k**, **Mean Reciprocal Rank (MRR)**, and automated **Citation Hallucination Check**.

---

## 📁 Deliverable Directory Structure

```
.
├── ingest/                 # Deliverable 1: Parsing, cleaning & section chunking
│   ├── clean_text.py       # Strips headers/footers/page numbers
│   ├── chunk_sections.py   # Parses statutory section boundaries
│   ├── spot_check.py       # Spot-checks key sections (26, 27, 33, 100, 108, 115, 120, 209)
│   └── run_ingest.py       # Linear ingestion runner
│
├── index/                  # Deliverable 2: Embeddings & FAISS index builder
│   ├── build_faiss_index.py# Builds FAISS IndexFlatIP + metadata JSON
│   └── run_index.py        # Indexing runner
│
├── rag/                    # Deliverable 3: RAG Orchestrator
│   ├── retriever.py        # Dense FAISS + Section-Number Regex Override
│   ├── generator.py        # Grounded LLM generation & prompt template
│   └── orchestrator.py     # Pipeline orchestrator & CLI interface
│
├── app/                    # Deliverable 4: Chatbot Interface
│   └── app.py              # Streamlit chat interface with statutory text viewer
│
├── eval/                   # Deliverable 5: Evaluation Harness
│   ├── test_set.json       # Hand-labelled benchmark test cases
│   ├── metrics.py          # Recall@k, MRR & Citation Hallucination checks
│   ├── eval_harness.py     # Test suite executor & report builder
│   └── run_eval.py         # Evaluation runner
│
├── data/
│   ├── raw/                # Source PDFs and HTML
│   ├── processed/          # chunks.jsonl, faiss_index.bin, faiss_metadata.json
│   └── eval/               # eval_report.md, results.csv
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup and Installation

### 1. Requirements
Ensure **Python 3.11+** is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and set your preferred `LLM_PROVIDER` and API key:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
*(Also supports `gemini`, `openai`, or `groq`)*

---

## 🚀 How to Run Stage-by-Stage

### Step 1: Ingestion Pipeline (`/ingest`)
Parses source legal PDFs, strips page headers/footers, chunks text strictly along legal section boundaries, saves `data/processed/chunks.jsonl`, and performs spot-checking on key sections:
```bash
python ingest/run_ingest.py
```

### Step 2: Build FAISS Vector Index (`/index`)
Generates 384-dimensional dense embeddings for all statutory section chunks and saves an in-process FAISS index binary (`data/processed/faiss_index.bin`) and metadata store (`data/processed/faiss_metadata.json`):
```bash
python index/run_index.py
```

### Step 3: Test RAG Orchestrator CLI (`/rag`)
Execute a query using the orchestrator (Section-Number Override + FAISS Dense Search + LLM Generation):
```bash
python -m rag.orchestrator "What is the notice period for terminating a permanent worker under Section 27?"
```

### Step 4: Run Evaluation Harness (`/eval`)
Evaluates the pipeline across the benchmark test set, computing **Recall@k**, **MRR**, and **Citation Hallucination Scores**, generating `data/eval/eval_report.md` and `data/eval/results.csv`:
```bash
python eval/run_eval.py
```

### Step 5: Launch Streamlit Chatbot UI (`/app`)
Run the interactive Streamlit chatbot application:
```bash
streamlit run app/app.py
```

---

## 📊 Benchmark Evaluation Results

| Metric | Target | Result | Description |
| :--- | :---: | :---: | :--- |
| **Recall@1** | > 80% | **100.0%** | Ground-truth section retrieved at rank 1 |
| **Recall@3** | > 90% | **100.0%** | Ground-truth section included in top-3 candidates |
| **Recall@5** | > 95% | **100.0%** | Ground-truth section included in top-5 candidates |
| **MRR (Mean Reciprocal Rank)** | > 0.85 | **1.0000** | Average reciprocal rank of target legal section |
| **Citation Hallucination Check** | 100% | **100.0%** | Verification that cited section is grounded in retrieved text |

---

## ⚖️ Legal Disclaimer
This chatbot provides informational guidance based strictly on the statutory text of the Bangladesh Labour Act 2006. It does not constitute official legal counsel.
