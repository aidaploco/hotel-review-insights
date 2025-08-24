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

from config import (
    GOOGLE_API_KEY,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL_NAME,
    LLM_MODEL_TYPE,
    LLM_TEMPERATURE,
    RETRIEVER_K
)

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

    Args:
        vector_store (Chroma): The vector store containing embedded hotel reviews.
        llm_type (str, optional): The type of LLM to use ("gemini" or "huggingface").
        llm_name (str, optional): The model name for the LLM ("gemini-2.0-flash" or "google/flan-t5-base").
        temperature (float, optional): LLM temperature setting (default: 0.7).
        max_output_tokens (int, optional): Maximum tokens for LLM output (default: 800).
        retriever_k (int, optional): Number of documents to retrieve per query (default: 5).
        google_api_key (str, optional): API key for Gemini models.
    """
    def __init__(
        self,
        vector_store: Chroma,
        llm_type: str = LLM_MODEL_TYPE,
        llm_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        retriever_k: int = RETRIEVER_K,
        google_api_key: str = GOOGLE_API_KEY
    ):
        self.vector_store = vector_store
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retriever_k = retriever_k
        self.google_api_key = google_api_key

        self.llm = self._initialize_llm()
        self.retrieval_chain = self._initialize_retrieval_chain()

    def _initialize_llm(self):
        """
        Initializes the appropriate LLM (Gemini or Hugging Face) based on LLM_MODEL_TYPE.
        """
        if self.llm_type.lower() == "gemini":
            if not self.google_api_key:
                logger.error("GOOGLE_API_KEY is missing for Gemini LLM initialization.")
                raise ValueError("GOOGLE_API_KEY is required for Gemini models.")
            logger.info(f"Initializing Gemini LLM: {self.llm_name}")
            return ChatGoogleGenerativeAI(
                model=self.llm_name,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                google_api_key=self.google_api_key,
            )
        elif self.llm_type.lower() == "huggingface":
            if not HUGGING_FACE:
                logger.error("Hugging Face 'transformers' or 'torch' not installed.")
                raise ImportError("Hugging Face 'transformers' or 'torch' not installed.")
            logger.info(f"Initializing Hugging Face LLM: {self.llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_name)
            llm_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1,
            )
            return HuggingFacePipeline(pipeline=llm_pipeline)
        else:
            logger.error(f"Unsupported LLM_MODEL_TYPE: {self.llm_type}")
            raise ValueError(f"Unsupported LLM_MODEL_TYPE: {self.llm_type}. Choose 'gemini' or 'huggingface'.")

    def _initialize_retrieval_chain(self):
        """
        Initializes the LangChain retrieval chain using LCEL.
        """
        # Define the prompt template for combining documents
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the user's question based on the provided context.
             If you don't know the answer, just say that you don't know, don't try to make up an answer.
             Keep the answer as concise as possible.\n\nContext: {context}"""),
            ("user", "{input}")
        ])

        # Create the document combining chain (stuff documents into prompt)
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create a retriever from the vector store
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.retriever_k})

        # Create the full retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    # Apply tenacity for robust API calls
    @retry(
        stop=stop_after_attempt(5),  # Try up to 5 times
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4s, 8s, etc., up to 10s max
        # Retry on any requests exception (including 429, 5xx)
        retry=(retry_if_exception_type(requests.exceptions.RequestException) |
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
