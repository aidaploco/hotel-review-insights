from unittest.mock import MagicMock, patch

import pytest
import requests
import tenacity

from src.rag_agent import HotelReviewRAGAgent


@pytest.fixture
def mock_vector_store():
    """Mocks a ChromaDB vector store."""
    mock_chroma = MagicMock()
    mock_retriever = MagicMock()
    mock_chroma.as_retriever.return_value = mock_retriever
    mock_retriever.similarity_search.return_value = [
        MagicMock(page_content="Relevant review snippet 1.", metadata={"hotel_name": "Grand Hotel"}),
        MagicMock(page_content="Relevant review snippet 2.", metadata={"hotel_name": "Cozy Inn"})
    ]
    return mock_chroma

@pytest.fixture
def mock_llm():
    """Mocks the LLM (ChatGoogleGenerativeAI or HuggingFacePipeline)."""
    mock_llm_instance = MagicMock()
    # LangChain's chat models expect a BaseMessage with a 'content' attribute.
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Mocked LLM response based on context."
    mock_llm_instance.invoke.return_value = mock_llm_response
    return mock_llm_instance

@pytest.fixture
def mock_retrieval_chain():
    """Mocks the LangChain retrieval chain."""
    mock_chain = MagicMock()
    # Ensure the mock has an 'invoke' method for patching
    mock_chain.invoke = MagicMock()
    return mock_chain

@pytest.fixture(autouse=True)
def patch_agent_initialization(mock_llm, mock_retrieval_chain):
    """
    Patches the _initialize_llm and _initialize_retrieval_chain methods of HotelReviewRAGAgent
    to return our mock instances.
    """
    with patch('src.rag_agent.HotelReviewRAGAgent._initialize_llm', return_value=mock_llm), \
         patch('src.rag_agent.HotelReviewRAGAgent._initialize_retrieval_chain', return_value=mock_retrieval_chain):
        yield mock_retrieval_chain # Yield the mock_retrieval_chain so tests can configure it


def test_rag_agent_initialization(mock_vector_store):
    """Tests that the RAG agent initializes correctly."""
    agent = HotelReviewRAGAgent(vector_store=mock_vector_store)
    assert agent.vector_store == mock_vector_store
    assert agent.llm is not None
    assert agent.retrieval_chain is not None

def test_ask_question_success(mock_vector_store, patch_agent_initialization):
    """Tests the ask_question method for a successful response."""
    agent = HotelReviewRAGAgent(mock_vector_store)
    question = "What about the staff?"

    mock_invoke = patch_agent_initialization.invoke
    mock_invoke.return_value = {
        "answer": "The staff were generally helpful and friendly.",
        "context": [
            MagicMock(page_content="Staff was very polite.", metadata={"hotel_name": "Hotel Alpha"}),
            MagicMock(page_content="Friendly reception.", metadata={"hotel_name": "Hotel Beta"})
        ]
    }

    response = agent.ask_question(question)

    assert response["query"] == question
    assert response["result"] == "The staff were generally helpful and friendly."
    assert len(response["source_documents"]) == 2
    assert "Hotel Alpha" in response["source_documents"][0].metadata["hotel_name"]

    mock_invoke.assert_called_once_with({"input": question})

def test_ask_question_no_answer(mock_vector_store, patch_agent_initialization):
    """Tests ask_question when the LLM provides no answer."""
    agent = HotelReviewRAGAgent(mock_vector_store)
    question = "Irrelevant question?"

    mock_invoke = patch_agent_initialization.invoke
    mock_invoke.return_value = {
        "answer": "No answer generated.",
        "context": []
    }

    response = agent.ask_question(question)

    assert response["query"] == question
    assert response["result"] == "No answer generated."
    assert len(response["source_documents"]) == 0
    mock_invoke.assert_called_once_with({"input": question})


def test_ask_question_api_error(mock_vector_store, patch_agent_initialization):
    """Tests ask_question with an API error, expecting retry."""
    agent = HotelReviewRAGAgent(mock_vector_store)
    question = "Question causing error?"

    mock_invoke = patch_agent_initialization.invoke
    mock_invoke.side_effect = requests.exceptions.RequestException("Mock API error")

    # Expect the call to raise tenacity.RetryError, as tenacity re-raises its own error
    with pytest.raises(tenacity.RetryError):
        agent.ask_question(question)

    # Verify that the invoke method was called multiple times due to retries
    assert mock_invoke.call_count >= 1
