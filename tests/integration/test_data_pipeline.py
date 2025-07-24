import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    KAGGLE_CSV_FILENAME,
    KAGGLE_DATASET_HANDLE,
)
from data_preprocessing import create_and_populate_vector_store, load_and_preprocess_data


@pytest.fixture(scope="module")
def dummy_downloaded_csv(tmp_path_factory):
    """
    Creates a dummy CSV file in a temporary directory to simulate a KaggleHub download.
    This CSV will be used as the source for load_and_preprocess_data.
    """
    download_dir = tmp_path_factory.mktemp("kaggle_download")
    csv_file_path = download_dir / KAGGLE_CSV_FILENAME

    data = {
        "Hotel_Name": ["Grand Hotel", "City Inn", "Luxury Stay"],
        "Positive_Review": ["Excellent service.", "Great location!", "Clean and comfy."],
        "Negative_Review": ["Noisy street.", "", "Small room."],
        "Reviewer_Nationality": ["USA", "Germany", "France"],
        "Review_Date": ["2023-01-01", "2023-01-05", "2023-01-10"],
        "Review_Total_Negative_Word_Counts": [2, 0, 2],
        "Review_Total_Positive_Word_Counts": [2, 2, 3],
        "Total_Number_of_Reviews": [100, 200, 150]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

    return str(download_dir)

@pytest.fixture(autouse=True)
def cleanup_chroma_db():
    """Removes the ChromaDB directory before and after each test."""
    test_db_path = os.path.join(CHROMA_DB_PATH, "test_integration_collection")
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    yield
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)


def test_full_data_pipeline_to_vector_store(dummy_downloaded_csv):
    """
    Verifies that load_and_preprocess_data correctly feeds into
    create_and_populate_vector_store and a real ChromaDB is populated.
    """
    with patch('data_preprocessing.kagglehub.dataset_download', return_value=dummy_downloaded_csv) \
          as mock_kaggle_download:
        test_chroma_db_path = os.path.join(CHROMA_DB_PATH, "test_integration_collection")

        raw_df = load_and_preprocess_data(
            data_dir="temp_data_dir",
            kaggle_dataset_handle=KAGGLE_DATASET_HANDLE,
            kaggle_csv_filename=KAGGLE_CSV_FILENAME
        )

        vector_store = create_and_populate_vector_store(
            raw_df,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            chroma_db_path=test_chroma_db_path,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # 1. Verify download was attempted
        mock_kaggle_download.assert_called_once_with(KAGGLE_DATASET_HANDLE)

        # 2. Verify DataFrame is correctly loaded and processed
        assert not raw_df.empty
        assert "full_review" in raw_df.columns
        assert len(raw_df) == 3

        # 3. Verify ChromaDB was created/populated and is accessible
        assert vector_store is not None
        assert os.path.exists(test_chroma_db_path)

        retrieved_count = vector_store._collection.count()
        assert retrieved_count > 0
