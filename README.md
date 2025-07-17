# Hotel Review Insights

An LLM & Agents project for extracting insights and answering questions from hotel reviews using Retrieval-Augmented Generation (RAG).

🌟 Overview
This project leverages large language models (LLMs) and a Retrieval-Augmented Generation (RAG) architecture to provide intelligent insights and answer specific questions based on a dataset of hotel reviews. By combining the power of semantic search with generative AI, it offers a powerful tool for understanding customer sentiment, identifying common issues, and extracting key information from vast amounts of unstructured text data.

✨ Features
Data Ingestion: Automatically downloads a large dataset of hotel reviews from KaggleHub.

Data Preprocessing: Cleans and combines positive/negative reviews into comprehensive review texts.

Vector Store Creation: Chunks review texts and generates embeddings using a Hugging Face Sentence Transformer model, storing them in a persistent ChromaDB vector store for efficient retrieval.

Retrieval-Augmented Generation (RAG): Integrates with either Google Gemini or Hugging Face models to answer user questions by first retrieving relevant review snippets from the vector store.

Robustness: Implements retry mechanisms for API calls using tenacity to handle transient network issues.

🚀 Technologies Used
Python 3.12+

uv: Fast Python package installer and resolver
langchain: Framework for developing applications powered by language models
chromadb: Open-source embedding database
pandas: Data manipulation and analysis
tenacity: Retrying library for robust API calls
transformers: Hugging Face's state-of-the-art NLP library
torch: PyTorch deep learning framework (for Hugging Face models)

🛠️ Configure Environment Variables

Create a .env file in the root of your project (next to pyproject.toml) and add your Google API Key and desired LLM configuration:

GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY" (Required if LLM_MODEL_TYPE is 'gemini')
LLM_MODEL_TYPE="gemini" (Options: "gemini" or "huggingface")
LLM_MODEL_NAME="gemini-2.0-flash" (For Gemini: "gemini-2.0-flash", "gemini-1.5-flash", etc. For Hugging Face: "google/flan-t5-small", etc.)
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" (Recommended for embeddings)

📁 Project Structure
.
├── .env                  Environment variables (ignored by Git)
├── .gitignore            Specifies intentionally untracked files to ignore
├── pyproject.toml        Project metadata and dependencies (managed by uv)
├── src/
│   ├── __init__.py       Makes 'src' a Python package
│   ├── config.py         Centralized configuration settings
│   ├── data_preprocessing.py Handles data loading, cleaning, and vector store creation
│   ├── rag_agent.py      Implements the RAG logic with LLM and retrieval chain
│   └── main.py           Main entry point for the application
└── tests/                Unit and integration tests
    ├── __init__.py
    ├── test_config.py
    ├── test_data_preprocessing.py
    └── test_rag_agent.py

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
