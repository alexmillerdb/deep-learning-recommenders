"""
Learning From Sets Dataset Processing Module

This module handles downloading, processing, and storing the Learning From Sets 
recommendation dataset. It provides functionality to:
1. Download and extract dataset files from the web
2. Process and transform data into suitable formats for recommender models
3. Save processed data to Unity Catalog and/or MDS format for training

The dataset contains movie ratings where each user has rated multiple movies.
"""
import sys
import os
import argparse
import subprocess
import pandas as pd
from databricks.connect import DatabricksSession
from typing import Dict, List, Tuple, Optional

# Set up import paths
current_dir = os.getcwd()
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to path

try:
    # Try importing the utility function
    from utils.path_utils import setup_import_paths
    setup_import_paths(current_dir, verbose=True)
except ImportError:
    # Fallback to manual path setup if the utility isn't available yet
    print("Warning: utils.path_utils not found, using manual path setup")
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(parent_dir, "utils"))
    sys.path.append(os.path.dirname(parent_dir))

from utils.data_processing import (
    get_dataset_statistics, index_string_columns, 
    binarize_ratings, split_dataset
)
from utils.storage import save_dataframe_to_mds


def run_shell_command(command: str) -> Optional[str]:
    """
    Execute shell commands safely and return output.
    
    Args:
        command: Shell command to execute
        
    Returns:
        Command output or None if failed
    
    Example:
        >>> run_shell_command("ls -la")
        # Returns directory listing or None if command fails
    """
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None


def download_and_extract_dataset(url: str, output_file: str) -> bool:
    """
    Download and extract dataset from URL.
    
    Args:
        url: URL to download from
        output_file: Filename to save downloaded content
        
    Returns:
        True if successful, False otherwise
    """
    import requests
    import zipfile
    import io
    
    try:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Download failed with status code {response.status_code}")
            return False
        
        # Save the zip file
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        return True
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        return False


def process_and_save_recommendation_data(
    df, user_col="userId", item_col="movieId", rating_col="rating", 
    output_paths=None, split_ratios=[0.7, 0.2, 0.1],
    index_columns=True, binarize=True, save_to_mds=True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline for processing recommendation data and saving to MDS.
    
    This function handles the entire workflow for preparing recommendation data:
    1. Computing dataset statistics
    2. Converting string columns to integer indices if necessary
    3. Selecting and renaming relevant columns
    4. Binarizing ratings (optional)
    5. Splitting data into train/validation/test sets
    6. Saving processed data to MDS format (optional)
    
    Args:
        df: Input Spark DataFrame
        user_col: User ID column name
        item_col: Item ID column name
        rating_col: Rating column name
        output_paths: Dictionary with paths for train/validation/test data
        split_ratios: Ratios for train/validation/test split
        index_columns: Whether to index string columns
        binarize: Whether to binarize ratings
        save_to_mds: Whether to save to MDS format
        
    Returns:
        Tuple of (train_df, validation_df, test_df)
    
    Example:
        >>> output_paths = {'train': '/path/to/train', 'validation': '/path/to/val', 'test': '/path/to/test'}
        >>> train_df, val_df, test_df = process_and_save_recommendation_data(
        ...     df, output_paths=output_paths, binarize=True
        ... )
    """
    # Print dataset statistics
    stats = get_dataset_statistics(df)
    print(f"Dataset statistics: {stats}")
    
    # Process columns that need indexing
    if index_columns:
        cols_to_index = []
        if df.schema[user_col].dataType.typeName() == 'string':
            cols_to_index.append(user_col)
        if df.schema[item_col].dataType.typeName() == 'string':
            cols_to_index.append(item_col)
            
        if cols_to_index:
            print(f"Indexing string columns: {cols_to_index}")
            df = index_string_columns(df, cols_to_index)
    
    # Select relevant columns
    df = df.select(user_col, item_col, rating_col)
    
    # Binarize ratings if requested
    if binarize:
        print(f"Binarizing ratings column: {rating_col}")
        df = binarize_ratings(df, rating_col=rating_col, output_col="label")
    else:
        df = df.withColumnRenamed(rating_col, "label")
    
    # Split dataset
    print(f"Splitting dataset with ratios: {split_ratios}")
    train_df, validation_df, test_df = split_dataset(df, ratios=split_ratios)
    
    # Print split sizes
    print(f"Training dataset: {train_df.count()} rows")
    print(f"Validation dataset: {validation_df.count()} rows")
    print(f"Test dataset: {test_df.count()} rows")
    
    # Save to MDS if requested
    if save_to_mds and output_paths:
        print("Preparing to save datasets to MDS format...")
        
        # Define column types for MDS
        cols = [user_col, item_col]
        cat_dict = {key: 'int64' for key in cols}
        label_dict = {'label': 'int'}
        columns_spec = {**label_dict, **cat_dict}
        
        # Save each dataset
        if 'train' in output_paths:
            print(f"Saving training data to: {output_paths['train']}")
            save_dataframe_to_mds(train_df, output_paths['train'], columns_spec)
        
        if 'validation' in output_paths:
            print(f"Saving validation data to: {output_paths['validation']}")
            save_dataframe_to_mds(validation_df, output_paths['validation'], columns_spec)
        
        if 'test' in output_paths:
            print(f"Saving test data to: {output_paths['test']}")
            save_dataframe_to_mds(test_df, output_paths['test'], columns_spec)
    
    return train_df, validation_df, test_df


def main():
    """
    Main entry point for the Learning From Sets dataset processing script.
    
    Handles command-line arguments, downloads the dataset, processes it,
    and saves it to the specified locations.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process Learning From Sets recommendation dataset and save to Unity Catalog'
    )
    parser.add_argument(
        '--catalog', 
        type=str, 
        required=True, 
        help='Catalog name in Unity Catalog'
    )
    parser.add_argument(
        '--schema', 
        type=str, 
        required=True, 
        help='Schema name in Unity Catalog'
    )
    parser.add_argument(
        '--save_data_to_uc_volumes', 
        action='store_true', 
        default=True, 
        help='Whether to save data to UC volumes'
    )
    parser.add_argument(
        '--output_dir_train', 
        type=str, 
        help='Output directory for training data in UC volumes'
    )
    parser.add_argument(
        '--output_dir_validation', 
        type=str, 
        help='Output directory for validation data in UC volumes'
    )
    parser.add_argument(
        '--output_dir_test', 
        type=str, 
        help='Output directory for test data in UC volumes'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=100000, 
        help='Limit the number of rows to process (default: 100000)'
    )
    args = parser.parse_args()

    # Download and extract the dataset
    dataset_url = "https://files.grouplens.org/datasets/learning-from-sets-2019/learning-from-sets-2019.zip"
    if not download_and_extract_dataset(dataset_url, "learning-from-sets-2019.zip"):
        print("Failed to download or extract dataset")
        return

    # Initialize Spark session
    print("Initializing Databricks Connect session...")
    # spark = DatabricksSession.builder.getOrCreate()
    
    try:
        # Load the CSV file into a pandas DataFrame
        print("Loading dataset from CSV...")
        df = pd.read_csv("learning-from-sets-2019/item_ratings.csv")
        
        # Create a Spark DataFrame from the pandas DataFrame and save it to UC
        print("Converting to Spark DataFrame...")
        spark_df = spark.createDataFrame(df)
        
        # Save to Unity Catalog
        table_name = f"{args.catalog}.{args.schema}.learning_from_sets_dataset"
        print(f"Saving full dataset to Unity Catalog: {table_name}")
        spark_df.write.mode("overwrite").saveAsTable(table_name)
        
        # Load the table and order by userId and movieId
        spark_df = spark.table(table_name)
        print(f"Full dataset size: {spark_df.count()} rows")
        
        # Order by userId and movieId for better representation
        ordered_df = spark_df.orderBy("userId", "movieId").limit(args.limit)
        print(f"Limited dataset size for processing: {ordered_df.count()} rows")
        
        # Process data if saving to UC volumes is enabled
        if args.save_data_to_uc_volumes:
            print("Processing data for MDS format...")
            
            # Validate required paths
            if not all([args.output_dir_train, args.output_dir_validation, args.output_dir_test]):
                print("Error: All output directories must be specified when saving to UC volumes")
                return
            
            output_paths = {
                'train': args.output_dir_train,
                'validation': args.output_dir_validation,
                'test': args.output_dir_test
            }
            
            process_and_save_recommendation_data(
                ordered_df,
                user_col="userId", 
                item_col="movieId",
                rating_col="rating",
                output_paths=output_paths,
                save_to_mds=True
            )
        
        print("Processing completed successfully")
            
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback

if __name__ == "__main__":
    main()