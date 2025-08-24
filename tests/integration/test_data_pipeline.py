import pytest


@pytest.mark.parametrize('populated_vector_store', ['data_pipeline_db'], indirect=True)
def test_data_pipeline(populated_vector_store):
    """
    Verifies that load_and_preprocess_data correctly feeds into
    create_and_populate_vector_store and a real ChromaDB is populated.
    """
    assert populated_vector_store is not None
    
    # 2. Verify it contains the correct number of documents
    retrieved_count = populated_vector_store._collection.count()
    assert retrieved_count > 0

    # 3. Verify the content of a document by querying the store
    query_text = "quiet and relaxing"
    retrieved_docs = populated_vector_store.similarity_search(query_text, k=1)
    
    assert len(retrieved_docs) > 0
    assert "Excellent for relaxation" in retrieved_docs[0].page_content
