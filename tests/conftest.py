import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, KAGGLE_CSV_FILENAME, KAGGLE_DATASET_HANDLE
from src.data_preprocessing import create_and_populate_vector_store, load_and_preprocess_data


@pytest.fixture
def dummy_csv_for_rag(tmp_path_factory):
    """
    Creates a dummy CSV file in a temporary directory to simulate a KaggleHub download.
    This fixture is shared by both integration tests.
    """
    download_dir = tmp_path_factory.mktemp("kaggle_download")
    csv_file_path = download_dir / KAGGLE_CSV_FILENAME

    data = {
        "Hotel_Name": ["Hotel Zen", "Urban Retreat", "Quiet Haven", "Hotel Alpha"],
        "Positive_Review": [
            "Very quiet and comfortable room. Excellent for relaxation.",
            "Great location, close to all main attractions. Staff were super helpful.",
            "Clean and spacious, though a bit far from the city center.",
            "Friendly reception and quick check-in. Rooms were spotless."
        ],
        "Negative_Review": [
            "Food was a bit pricey.",
            "Bathroom was quite small.",
            "Some noise from outside traffic at night.",
            "The AC was a bit loud."
        ],
        "Reviewer_Nationality": ["USA", "UK", "Canada", "Germany"],
        "Review_Date": ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"],
        "Review_Total_Negative_Word_Counts": [4, 4, 6, 5],
        "Review_Total_Positive_Word_Counts": [6, 7, 5, 5],
        "Total_Number_of_Reviews": [100, 200, 150, 120]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

    return str(download_dir)

@pytest.fixture
def populated_vector_store(dummy_csv_for_rag, tmp_path_factory, request):
    """
    Loads data and populates a real ChromaDB instance using actual embeddings.
    This fixture is shared by both integration tests.
    """
    db_name = getattr(request, 'param', 'db_path')
    test_db_path = tmp_path_factory.mktemp(db_name)
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

    with patch('src.data_preprocessing.kagglehub.dataset_download', return_value=dummy_csv_for_rag):
        temp_data_load_dir = tmp_path_factory.mktemp("temp_data_load_dir")
        raw_df = load_and_preprocess_data(
            data_dir=str(temp_data_load_dir),
            kaggle_dataset_handle=KAGGLE_DATASET_HANDLE,
            kaggle_csv_filename=KAGGLE_CSV_FILENAME
        )

        vector_store = create_and_populate_vector_store(
            raw_df,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            chroma_db_path=str(test_db_path),
            collection_name=CHROMA_COLLECTION_NAME
        )
        yield vector_store

    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
