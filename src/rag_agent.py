import logging
import os
from typing import Any, Dict

import requests

# Langchain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import GOOGLE_API_KEY, LLM_MAX_OUTPUT_TOKENS, LLM_MODEL_NAME, LLM_MODEL_TYPE, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# Hugging Face specific imports
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    HUGGING_FACE = True
except ImportError:
    HUGGING_FACE = False
    logging.warning("""Hugging Face 'transformers' or 'torch' not found.
                    Hugging Face models will not be available for RAG.""")


class HotelReviewRAGAgent:
    """
    An agent that performs Retrieval-Augmented Generation (RAG) on hotel reviews.
    It retrieves relevant review snippets from a vector store and uses an LLM
    to answer questions based on the retrieved context.
    """
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = self._initialize_llm()
        self.retrieval_chain = self._initialize_retrieval_chain()
        logger.info(f"HotelReviewRAGAgent initialized with {LLM_MODEL_TYPE} LLM and new retrieval chain.")

    def _initialize_llm(self):
        """
        Initializes the appropriate LLM (Gemini or Hugging Face) based on LLM_MODEL_TYPE.
        """
        if LLM_MODEL_TYPE.lower() == "gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for Gemini models.")
            logger.info(f"Initializing Gemini LLM: {LLM_MODEL_NAME}")
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                google_api_key=GOOGLE_API_KEY,
            )
        elif LLM_MODEL_TYPE.lower() == "huggingface":
            if not HUGGING_FACE:
                raise ImportError("""Hugging Face 'transformers' or 'torch' not installed.
                                  Cannot initialize Hugging Face LLM.""")
            logger.info(f"Initializing Hugging Face LLM: {LLM_MODEL_NAME}")
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

            # Create a Hugging Face pipeline for text generation
            llm_pipeline = pipeline(
                "text2text-generation",  # Suitable for Flan-T5
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=LLM_MAX_OUTPUT_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            )
            llm = HuggingFacePipeline(pipeline=llm_pipeline)
        else:
            raise ValueError(f"Unsupported LLM_MODEL_TYPE: {LLM_MODEL_TYPE}. Choose 'gemini' or 'huggingface'.")
        return llm

    def _initialize_retrieval_chain(self):
        """
        Initializes the LangChain retrieval chain using LCEL.
        """
        # 1. Define the prompt template for combining documents
        # This prompt is for the LLM to generate an answer based on the retrieved context.
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the user's question based on the provided context.
             If you don't know the answer, just say that you don't know, don't try to make up an answer.
             Keep the answer as concise as possible.\n\nContext: {context}"""),
            ("user", "{input}")
        ])

        # 2. Create the document combining chain (stuff documents into prompt)
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 3. Create a retriever from the vector store
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

        # 4. Create the full retrieval chain
        # This chain takes the user's question, retrieves documents, and then passes them to the document_chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    # Apply tenacity for robust API calls
    @retry(
        stop=stop_after_attempt(5),  # Try up to 5 times
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4s, 8s, etc., up to 10s max
        retry=(retry_if_exception_type(requests.exceptions.RequestException) | # Retry on any requests exception (including 429, 5xx)
               retry_if_exception_type(ValueError)) # Also retry on ValueError, common for LangChain API errors
    )
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Asks a question to the RAG agent and returns the answer along with source documents.
        Includes retry logic for API calls.

        Args:
            question (str): The natural language question to ask.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and source documents.
                            Example: {"query": "...", "result": "...", "source_documents": [...]}
        """
        logger.info(f"Asking RAG agent: '{question}'")
        try:
            response = self.retrieval_chain.invoke({"input": question})
            answer = response.get("answer", "No answer generated.")
            source_documents = response.get("context", []) # Context contains the retrieved Document objects

            logger.info("RAG agent successfully generated response.")
            return {
                "query": question,
                "result": answer,
                "source_documents": source_documents
            }
        except Exception as e:
            logger.error(f"Error asking question to RAG agent (attempting retry if configured): {e}", exc_info=True)
            raise  # Re-raise to be caught by the tenacity decorator or main function's try-except


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # This block requires a pre-populated ChromaDB.
    # For a full test, run main.py first to ensure the DB is created.

    from config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME

    try:
        # Load the vector store
        # Need to ensure embeddings are initialized for Chroma to load/create
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = None

        if os.path.exists(CHROMA_DB_PATH):
            try:
                vector_store = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=embeddings,
                    collection_name=CHROMA_COLLECTION_NAME
                )
                if vector_store._collection.count() == 0:
                    logger.error("Existing ChromaDB is empty, run main.py first")
                else:
                    logger.info("Successfully loaded existing ChromaDB for RAG agent test.")
            except Exception as e:
                logger.error(f"Could not load existing ChromaDB for RAG test: {e}")
        else:
            logger.error("ChromaDB not found, run main.py first")

        if vector_store:
            agent = HotelReviewRAGAgent(vector_store=vector_store)

            # Example questions
            questions = [
                "What are the common complaints about staff in hotels?",
                "Tell me about positive experiences with hotel rooms.",
                "Are there any reviews mentioning issues with breakfast?",
                "What is the overall sentiment regarding hotel locations?",
                "Summarize reviews about cleanliness."
            ]

            for q in questions:
                try:
                    response = agent.ask_question(q)
                    logger.info(f"\n--- Question: {response['query']} ---")
                    logger.info(f"Answer: {response['result']}")
                    logger.info("Source Documents (truncated):")
                    for i, doc in enumerate(response["source_documents"]):
                        logger.info(f"""  Doc {i+1} (Hotel: {doc.metadata.get('hotel_name', 'N/A')},
                                    Nationality: {doc.metadata.get('reviewer_nationality', 'N/A')}):
                                    {doc.page_content[:200]}...""")
                except Exception as inner_e:
                    logger.error(f"Failed to get answer for question '{q}' after retries: {inner_e}")
        else:
            logger.error("Vector store could not be initialized for RAG agent test.")

    except Exception as e:
        logger.critical(f"An error occurred during RAG agent test: {e}", exc_info=True)
        logger.error("Ensure all required libraries are installed and .env variables are set correctly.")
