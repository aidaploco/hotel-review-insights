import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Data Paths ---
DATA_DIR = "data"
KAGGLE_DATASET_HANDLE = "jiashenliu/515k-hotel-reviews-data-in-europe"
KAGGLE_CSV_FILENAME = "Hotel_Reviews.csv"

# --- LLM Parameters ---
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "gemini")  # Define the type of model to use: "gemini" or "huggingface"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash") # For Hugging Face: "google/flan-t5-small"
LLM_TEMPERATURE = 0.7  # Controls randomness (0.0 for deterministic, 1.0 for very creative)
LLM_MAX_OUTPUT_TOKENS = 800  # Maximum tokens in the LLM's response

# --- Embedding Model Parameters ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- ChromaDB Parameters ---
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
CHROMA_COLLECTION_NAME = "hotel_reviews"

# --- Retrieval Parameters ---
RETRIEVER_K = 5
