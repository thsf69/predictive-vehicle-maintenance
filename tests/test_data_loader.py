import pytest
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data

def test_load_data_valid_file(tmp_path):
    # Create a dummy csv
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "data.csv"
    p.write_text("col1,col2\n1,2\n3,4")
    
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'col1' in df.columns

def test_load_data_invalid_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
