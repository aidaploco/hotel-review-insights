import os
import shutil
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from src.data_preprocessing import create_and_populate_vector_store, load_and_preprocess_data


@pytest.fixture
def dummy_csv(tmp_path):
    """Creates a dummy CSV file for testing data loading."""
    data = {
        "Hotel_Name": ["Hotel A", "Hotel B", "Hotel C"],
        "Positive_Review": ["Great staff.", "Good location.", ""],
        "Negative_Review": ["Noisy room.", "", "Bad breakfast."],
        "Reviewer_Nationality": ["USA", "UK", "Germany"],
        "Review_Date": ["1/1/2023", "1/2/2023", "1/3/2023"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "Hotel_Reviews.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def mock_embeddings():
    """Mocks HuggingFaceEmbeddings."""
    with patch('src.data_preprocessing.HuggingFaceEmbeddings') as mock_hf_embeddings:
        mock_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_chroma():
    """Mocks ChromaDB."""
    with patch('src.data_preprocessing.Chroma') as mock_chroma_class:
        mock_instance = MagicMock()
        mock_instance._collection.count.return_value = 0 # Default to empty collection for initial load check
        mock_chroma_class.return_value = mock_instance
        # Ensure Chroma.from_documents returns a mock that has a count > 0
        mock_from_documents_instance = MagicMock()
        mock_from_documents_instance._collection.count.return_value = 1 # Simulate documents added
        mock_chroma_class.from_documents.return_value = mock_from_documents_instance
        yield mock_instance, mock_chroma_class


def test_load_and_preprocess_data_local_file(dummy_csv):
    """Tests loading and preprocessing from a local CSV file."""
    # Mock kagglehub.dataset_download to ensure it's not called
    with patch('src.data_preprocessing.kagglehub.dataset_download') as mock_download:
        df = load_and_preprocess_data(data_dir=str(dummy_csv.parent), kaggle_csv_filename=dummy_csv.name)
        mock_download.assert_not_called() # Ensure download was not attempted

    assert not df.empty
    assert "full_review" in df.columns
    assert len(df) == 3
    assert df.loc[0, "full_review"] == "Great staff. Noisy room."
    assert df.loc[1, "full_review"] == "Good location."
    assert df.loc[2, "full_review"] == "Bad breakfast."

def test_load_and_preprocess_data_kaggle_download(tmp_path):
    """Tests loading and preprocessing when data needs to be downloaded from KaggleHub."""
    # Simulate kagglehub download by returning a path to our dummy CSV
    with patch('src.data_preprocessing.kagglehub.dataset_download', return_value=str(tmp_path)) as mock_download:
        # Create the dummy CSV in the mocked download path
        data = {
            "Hotel_Name": ["Hotel X"],
            "Positive_Review": ["Clean rooms."],
            "Negative_Review": ["Small bathroom."],
            "Reviewer_Nationality": ["France"],
            "Review_Date": ["2/1/2023"]
        }
        df_dummy = pd.DataFrame(data)
        csv_path_in_download = tmp_path / "Hotel_Reviews.csv"
        df_dummy.to_csv(csv_path_in_download, index=False)

        # Call the function, expecting it to trigger download
        df = load_and_preprocess_data(data_dir=str(tmp_path / "my_data_dir"), kaggle_csv_filename="Hotel_Reviews.csv")

        mock_download.assert_called_once() # Ensure download was attempted
        assert not df.empty
        assert len(df) == 1
        assert df.loc[0, "full_review"] == "Clean rooms. Small bathroom."

def test_load_and_preprocess_data_empty_reviews(tmp_path):
    """Tests handling of reviews that become empty after preprocessing."""
    data = {
        "Hotel_Name": ["Hotel A", "Hotel B"],
        "Positive_Review": ["", ""],
        "Negative_Review": ["", ""],
        "Reviewer_Nationality": ["USA", "UK"],
        "Review_Date": ["1/1/2023", "1/2/2023"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "empty_reviews.csv"
    df.to_csv(csv_path, index=False)

    with patch('src.data_preprocessing.kagglehub.dataset_download'):
        processed_df = load_and_preprocess_data(data_dir=str(tmp_path), kaggle_csv_filename="empty_reviews.csv")
        assert processed_df.empty

def test_create_and_populate_vector_store_new(tmp_path, mock_embeddings, mock_chroma):
    """Tests creating and populating a new vector store."""
    _, mock_chroma_class = mock_chroma

    df = pd.DataFrame({
        "Hotel_Name": ["Hotel A"],
        "Reviewer_Nationality": ["USA"],
        "Review_Date": ["1/1/2023"],
        "full_review": ["This is a test review."]
    })

    # Ensure the chroma_db_path does not exist initially for this test
    chroma_path = tmp_path / "test_chroma_db_new"
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    vector_store = create_and_populate_vector_store(
        df,
        chroma_db_path=str(chroma_path),
        collection_name="test_collection"
    )

    # Assert from_documents was called with correct args
    mock_chroma_class.from_documents.assert_called_once_with(
        documents=ANY, # Exact list of Document objects can be complex
        embedding=mock_embeddings,
        persist_directory=str(chroma_path), # Check that persist_directory was passed
        collection_name="test_collection"
    )
    assert vector_store is not None
    assert vector_store._collection.count() == 1

def test_create_and_populate_vector_store_load_existing(tmp_path, mock_embeddings, mock_chroma):
    """Tests loading an existing, populated vector store."""
    mock_chroma_instance, mock_chroma_class = mock_chroma
    mock_chroma_instance._collection.count.return_value = 5 # Simulate existing documents

    df = pd.DataFrame({ # This df won't be used if DB is loaded
        "Hotel_Name": ["Hotel A"],
        "Reviewer_Nationality": ["USA"],
        "Review_Date": ["1/1/2023"],
        "full_review": ["This is a test review."]
    })

    # Simulate chroma_db_path existing
    chroma_path = tmp_path / "test_chroma_db_existing"
    os.makedirs(chroma_path, exist_ok=True) # Create the directory

    vector_store = create_and_populate_vector_store(
        df,
        chroma_db_path=str(chroma_path),
        collection_name="test_collection"
    )

    mock_chroma_class.assert_called_once_with( # Should attempt to load existing Chroma with correct args
        persist_directory=str(chroma_path),
        embedding_function=mock_embeddings,
        collection_name="test_collection"
    )
    mock_chroma_class.from_documents.assert_not_called() # Should NOT call from_documents
    assert vector_store is not None
    assert vector_store._collection.count() == 5
