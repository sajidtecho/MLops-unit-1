"""
MLOps Unit 1 - Dataset Loading and Basic Statistics
This script demonstrates loading a dataset and computing basic statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_analyze_dataset(file_path):
    """
    Load a dataset from a CSV file and print basic statistics.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded dataset from: {file_path}")
        print(f"\n{'='*60}")
        print("DATASET OVERVIEW")
        print(f"{'='*60}")
        
        # Basic statistics
        print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn Names and Types:")
        print(df.dtypes)
        
        print(f"\n{'='*60}")
        print("DESCRIPTIVE STATISTICS")
        print(f"{'='*60}")
        print(df.describe())
        
        print(f"\n{'='*60}")
        print("MISSING VALUES")
        print(f"{'='*60}")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print(missing[missing > 0])
        
        print(f"\n{'='*60}")
        print("FIRST FEW ROWS")
        print(f"{'='*60}")
        print(df.head())
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print("Creating a sample dataset for demonstration...")
        return create_sample_dataset()

def compute_advanced_statistics(df):
    """Compute advanced statistics for numerical columns."""
    print(f"\n{'='*60}")
    print("ADVANCED STATISTICS (EXPERIMENT-V1)")
    print(f"{'='*60}")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Skewness: {df[col].skew():.4f}")
        print(f"  Kurtosis: {df[col].kurtosis():.4f}")
        print(f"  Variance: {df[col].var():.4f}")
        print(f"  Standard Error: {df[col].sem():.4f}")

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    sample_data = {
        'ID': np.arange(1, 101),
        'Age': np.random.randint(18, 80, 100),
        'Income': np.random.randint(20000, 150000, 100),
        'Score': np.random.uniform(0, 100, 100)
    }
    df = pd.DataFrame(sample_data)
    print("\n✓ Sample dataset created successfully")
    return df

if __name__ == "__main__":
    # Try to load the sample data from DVC directory
    data_path = Path(__file__).parent / "DVC" / "data" / "sample_data.csv"
    
    if data_path.exists():
        print(f"Loading dataset from: {data_path}\n")
        df = load_and_analyze_dataset(str(data_path))
    else:
        print(f"Data file not found at {data_path}")
        print("Creating and analyzing sample dataset...\n")
        df = create_sample_dataset()
        print(f"\n{'='*60}")
        print("DESCRIPTIVE STATISTICS")
        print(f"{'='*60}")
        print(df.describe())
    
    # Compute advanced statistics (Experiment-V1 Feature)
    compute_advanced_statistics(df)
    
    print("\n✓ Analysis complete!")
