import logging

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_TYPE
from src.data_preprocessing import create_and_populate_vector_store, load_and_preprocess_data
from src.rag_agent import HotelReviewRAGAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to orchestrate the hotel review RAG project.
    1. Loads and preprocesses the data.
    2. Creates/loads the ChromaDB vector store.
    3. Initializes the HotelReviewRAGAgent.
    4. Enters a loop to allow users to ask questions.
    """
    logger.info("Starting Hotel Review RAG project...")

    # 1. Load and Preprocess Data
    reviews_df = load_and_preprocess_data()
    if reviews_df.empty:
        logger.error("No reviews to process. Exiting.")
        return

    # Limit the number of reviews for quick testing/demonstration
    reviews_df = reviews_df.head(1000)
    logger.info(f"Preparing to process {len(reviews_df)} reviews for vector store population.")

    # 2. Create/Load the ChromaDB vector store
    try:
        vector_store = create_and_populate_vector_store(
            reviews_df,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME
        )
        if vector_store._collection.count() == 0:
            logger.error("Failed to create or load vector store. Exiting.")
            return
    except Exception as e:
        logger.critical(f"Error creating/loading ChromaDB vector store: {e}", exc_info=True)
        logger.error("Ensure 'sentence-transformers' is installed and you have sufficient disk space.")
        return

    # 3. Initialize the HotelReviewRAGAgent
    try:
        rag_agent = HotelReviewRAGAgent(vector_store=vector_store)
    except Exception as e:
        logger.critical(f"Error initializing HotelReviewRAGAgent: {e}", exc_info=True)
        logger.error("Ensure MODEL_TYPE and LLM_MODEL_NAME are correctly set in your .env file.")
        if LLM_MODEL_TYPE.lower() == "gemini":
            logger.error("Ensure GOOGLE_API_KEY is set for Gemini models.")
        elif LLM_MODEL_TYPE.lower() == "huggingface":
            logger.error("Ensure 'transformers' and 'torch' are installed for Hugging Face models.")
        return

    # 4. Enter a loop to allow users to ask questions
    logger.info("\n--- RAG Agent Ready ---")
    logger.info("You can now ask questions about the hotel reviews.")
    logger.info("Type 'exit' or 'quit' to end the session.")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            logger.info("Exiting RAG session. Goodbye!")
            break

        if not user_query.strip():
            logger.warning("Enter a non-empty question.")
            continue

        try:
            response = rag_agent.ask_question(user_query)

            logger.info("\n--- Answer ---")
            logger.info(f"Question: {response['query']}")
            logger.info(f"Answer: {response['result']}")

            if response["source_documents"]:
                logger.info("Source Documents Used:")
                for i, doc in enumerate(response["source_documents"]):
                    logger.info(f"""  Doc {i+1} (Hotel: {doc.metadata.get('hotel_name', 'N/A')}, Nationality:
                                {doc.metadata.get('reviewer_nationality', 'N/A')}): {doc.page_content[:200]}...""")
            else:
                logger.info("No source documents were retrieved for this query.")
        except Exception as e:
            logger.error(f"Failed to get answer for question '{user_query}' after all retries: {e}", exc_info=True)
            logger.info("Please try again or check your API key/model setup.")

    logger.info("\nHotel Review RAG project finished.")


if __name__ == "__main__":
    main()
