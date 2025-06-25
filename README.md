# 🧠 GenAI Product Data Intelligence using Retrieval-Augmented Generation (RAG)

A local LLM-powered RAG system to answer **complex natural language queries** over **industrial product datasets**, combining structured and unstructured information (product master, classification, pricing, and customer reviews).

---

## 📌 Use Case

Industrial product data is typically siloed across systems like:
- Item master data
- Product classifications
- Price catalogs
- Customer reviews (free-text)

Business users often ask:
> "What are the ACTIVE mechanical items with avg rating below 3 and shelf life info?"

This app enables semantic, cross-source Q&A using:
- 📄 CSV data ingestion
- 🧠 Embeddings + vector search
- 🤖 Local LLM-based answer generation (no external API)
- 🧾 Tabular and natural language answers

---

## 🛠️ Features

| Feature | Description |
|--------|-------------|
| 🔍 Natural Language Query | Ask business-level questions across datasets |
| 🧠 Retrieval-Augmented Generation (RAG) | Combines retrieval + local LLM for deep understanding |
| 💾 Local Vector Store | Uses ChromaDB for fast semantic retrieval |
| 📊 Post-processed Answers | Structured into readable tables (where possible) |
| 🎛️ Streamlit UI | Easy-to-use web app to explore queries + answers |

---

## 🗂️ Folder Structure

```bash
.
├── rag_pipeline/
│   ├── __init__.py
│   ├── retriever.py          # Embedding + retrieval from vector DB
│   ├── generator.py          # Prompt building + local LLM response
│   └── utils/
│       └── logger.py         # Logging support
├── data/
│   ├── item_master.csv
│   ├── item_classification.csv
│   ├── item_prices.csv
│   └── customer_reviews.csv
├── offload/                  # (Optional) for model offloading
├── app.py                    # Main Streamlit app
└── README.md

⚙️ Installation

- Clone the repository

`git clone https://github.com/your-username/genai-product-rag.git`
`cd genai-product-rag`

- Create & activates virtual environment
`python -m venv venv``
# Windows
`venv\Scripts\activate`
# macOS/Linux
`source venv/bin/activate`

-'Installs dependencies'
pip install -r requirements.txt

- Run the Streamlit app

`streamlit run app.py`

