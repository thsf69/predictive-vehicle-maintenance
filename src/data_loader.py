import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        df = pd.DataFrame(pd.read_csv(filepath))
        logger.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise e
