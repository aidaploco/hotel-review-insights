from importlib import reload
from unittest.mock import patch

import src.config as config


def test_google_api_key_loaded():
    """
    Test that GOOGLE_API_KEY is loaded from environment variables.
    """
    with patch('os.getenv', side_effect=lambda key, default=None: "test_api_key_123" \
               if key == "GOOGLE_API_KEY" else default):
        reload(config)
        assert config.GOOGLE_API_KEY == "test_api_key_123"

def test_data_paths_exist():
    """
    Test that data directory and Kaggle handle are correctly defined.
    """
    assert config.DATA_DIR == "data"
    assert config.KAGGLE_DATASET_HANDLE == "jiashenliu/515k-hotel-reviews-data-in-europe"

def test_set_model_type():
    """
    Test that LLM_MODEL_TYPE is correctly read if set.
    """
    with patch('os.getenv', side_effect=lambda key, default=None: "gemini" if key == "LLM_MODEL_TYPE" else default):
        reload(config)
        assert config.LLM_MODEL_TYPE == "gemini"

