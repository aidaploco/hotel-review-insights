import logging
import os

import kagglehub
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    KAGGLE_CSV_FILENAME,
    KAGGLE_DATASET_HANDLE,
)

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_dir: str = DATA_DIR, kaggle_dataset_handle: str = KAGGLE_DATASET_HANDLE,
                             kaggle_csv_filename: str = KAGGLE_CSV_FILENAME) -> pd.DataFrame:
    """
    Loads the raw hotel review dataset, combines positive and negative reviews,
    and performs basic cleaning. Attempts to download from KaggleHub if not found locally.

    Args:
        data_dir (str): The directory where the raw data file is located.
        kaggle_dataset_handle (str): The Kaggle dataset handle.
        kaggle_csv_filename (str): The name of the CSV file within the Kaggle dataset.

    Returns:
        pd.DataFrame: A DataFrame with combined and cleaned review text.
    """
    local_file_path = os.path.join(data_dir, kaggle_csv_filename)
    logger.info(f"Attempting to load data from: {local_file_path}")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_to_read = None

    if not os.path.exists(local_file_path):
        logger.info(f"Data file not found locally at {local_file_path}. Attempting to download from KaggleHub.")
        try:
            downloaded_path = kagglehub.dataset_download(kaggle_dataset_handle)
            downloaded_file_path = os.path.join(downloaded_path, kaggle_csv_filename)

            if os.path.exists(downloaded_file_path):
                file_to_read = downloaded_file_path
            else:
                logger.error(f"""Downloaded KaggleHub dataset did not contain '{kaggle_csv_filename}'
                             at expected path: {downloaded_file_path}""")
                raise FileNotFoundError(f"""KaggleHub download successful, but '{kaggle_csv_filename}'
                                        not found at {downloaded_file_path}""")

        except Exception as e:
            logger.error(f"Error downloading data from KaggleHub: {e}", exc_info=True)
            raise FileNotFoundError(f"Failed to download data from KaggleHub: {e}")
    else:
        file_to_read = local_file_path # Use local file if it exists

    if file_to_read is None: # Added check for file_to_read
        logger.error("No file path determined for reading data.")
        raise FileNotFoundError("Could not determine a valid file path for data loading.")

    try:
        df = pd.read_csv(file_to_read)
        logger.info(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading CSV from {file_to_read}: {e}", exc_info=True)
        raise Exception(f"Failed to load CSV from '{file_to_read}': {e}") from e

    # Handle NaN values by replacing them with empty strings before combining
    df["Positive_Review"] = df["Positive_Review"].fillna("")
    df["Negative_Review"] = df["Negative_Review"].fillna("")

    # Combine Positive_Review and Negative_Review into a single "full_review" column
    df["full_review"] = df["Positive_Review"] + " " + df["Negative_Review"]

    # Basic cleaning: remove extra spaces
    df["full_review"] = df["full_review"].str.strip().str.replace(r"\s+", " ", regex=True)

    # Filter out reviews that are empty after combination
    df = df[df["full_review"].str.len() > 0]
    logger.info(f"After combining and cleaning, {len(df)} non-empty reviews remain.")

    # Select relevant columns for further processing
    processed_df = df[["Hotel_Name", "Reviewer_Nationality", "Review_Date", "full_review"]].copy()

    return processed_df


def create_and_populate_vector_store(
    df: pd.DataFrame,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    chroma_db_path: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME
) -> Chroma:
    """
    Splits the review texts into chunks, generates embeddings, and populates a ChromaDB vector store.

    Args:
        df (pd.DataFrame): DataFrame containing "full_review" and metadata.
        embedding_model_name (str): Name of the Hugging Face embedding model.
        chroma_db_path (str): Path to store the ChromaDB persistent data.
        collection_name (str): Name of the collection within ChromaDB.

    Returns:
        Chroma: An initialized Chroma vector store.
    """
    logger.info("Initializing text splitter...")
    # Using RecursiveCharacterTextSplitter to maintain semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )

    logger.info(f"Loading embedding model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Prepare documents for LangChain
    documents = []
    for index, row in df.iterrows():
        # Create a dictionary for metadata
        metadata = {
            "review_id": index,
            "hotel_name": row["Hotel_Name"],
            "reviewer_nationality": row["Reviewer_Nationality"],
            "review_date": row["Review_Date"],
            "original_review_text_start": row["full_review"][:100] + "..." \
                if len(row["full_review"]) > 100 else row["full_review"]
        }
        # Split the full review text into smaller chunks
        chunks = text_splitter.create_documents([row["full_review"]], metadatas=[metadata])
        documents.extend(chunks)

    logger.info(f"Split {len(df)} reviews into {len(documents)} chunks.")

    # Check if ChromaDB already exists and has data for this collection
    if os.path.exists(chroma_db_path):
        try:
            # Attempt to load existing ChromaDB
            vector_store = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            # Check if the collection has any data
            if vector_store._collection.count() > 0:
                logger.info(f"""ChromaDB collection '{collection_name}' already exists with
                            {vector_store._collection.count()} documents. Skipping re-population.""")
                return vector_store
            else:
                logger.info(f"ChromaDB directory exists, but collection '{collection_name}' is empty. Populating...")
        except Exception as e:
            logger.warning(f"Could not load existing ChromaDB at {chroma_db_path}: {e}. Re-creating it.")
            # If loading fails, proceed to create a new one
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=chroma_db_path,
                collection_name=collection_name
            )
    else:
        logger.info(f"Creating new ChromaDB at {chroma_db_path} and populating with {len(documents)} chunks.")
        # Create a new ChromaDB and populate it
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path,
            collection_name=collection_name
        )

    logger.info("ChromaDB population complete and persisted to disk.")
    return vector_store


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Example usage:
    reviews_df = load_and_preprocess_data()
    if not reviews_df.empty:
        # Limit for quick testing
        reviews_df_sample = reviews_df.head(10)
        logger.info(f"Creating vector store with {len(reviews_df_sample)} sample reviews...")
        vector_store = create_and_populate_vector_store(reviews_df_sample)
        logger.info(f"Vector store created/loaded with {vector_store._collection.count()} documents.")

        # Example of querying the vector store
        query = "What are the common complaints about staff?"
        logger.info(f"\nSearching for documents related to: '{query}'")
        retrieved_docs = vector_store.similarity_search(query, k=3)
        logger.info("Retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"--- Document {i+1} ---")
            logger.info(f"Content: {doc.page_content[:200]}...")
            logger.info(f"Metadata: {doc.metadata}")
