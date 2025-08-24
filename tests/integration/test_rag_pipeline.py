from typing import Any

import pytest

from src.rag_agent import HotelReviewRAGAgent


@pytest.mark.parametrize('populated_vector_store', ['rag_pipeline_db'], indirect=True)
def test_rag_pipeline(populated_vector_store: Any):
    agent = HotelReviewRAGAgent(
        vector_store=populated_vector_store,
        llm_type="huggingface",
        llm_name="google/flan-t5-base"
    )

    question = "What can you tell me about the room at Hotel Zen?"
    response = agent.ask_question(question)

    assert isinstance(response["result"], str)
    assert len(response["result"]) > 0
    assert len(response["source_documents"]) > 0
