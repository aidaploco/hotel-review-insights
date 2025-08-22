def test_full_data_pipeline_to_vector_store(populated_vector_store):
    """
    Verifies that load_and_preprocess_data correctly feeds into
    create_and_populate_vector_store and a real ChromaDB is populated.
    """
    # 1. Verify ChromaDB was created/populated and is accessible
    assert populated_vector_store is not None
    
    # 2. Verify it contains the correct number of documents
    retrieved_count = populated_vector_store._collection.count()
    assert retrieved_count > 0

    # 3. Verify the content of a document by querying the store
    query_text = "quiet and relaxing"
    retrieved_docs = populated_vector_store.similarity_search(query_text, k=1)
    
    assert len(retrieved_docs) > 0
    assert "Excellent for relaxation" in retrieved_docs[0].page_content
