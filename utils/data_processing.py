from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType
from pyspark.ml.feature import StringIndexer
from typing import Dict, List, Tuple, Optional, Union


def get_dataset_statistics(df: DataFrame) -> Dict:
    """
    Get basic statistics about a dataset.
    
    Args:
        df: Input Spark DataFrame
        
    Returns:
        Dictionary containing dataset statistics
    """
    total_rows = df.count()
    
    # Count distinct values for each column
    stats = {"total_rows": total_rows}
    for col_name in df.columns:
        distinct_count = df.select(F.countDistinct(col_name)).collect()[0][0]
        stats[f"distinct_{col_name}"] = distinct_count
    
    return stats


def index_string_columns(df: DataFrame, cols_to_index: List[str]) -> DataFrame:
    """
    Convert string columns to numeric indices.
    
    Args:
        df: Input Spark DataFrame
        cols_to_index: List of column names to index
        
    Returns:
        DataFrame with indexed columns
    """
    result_df = df
    for col in cols_to_index:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        result_df = indexer.fit(result_df).transform(result_df)
        result_df = result_df.withColumn(col, result_df[f"{col}_index"].cast(LongType())).drop(f"{col}_index")
    
    return result_df


def binarize_ratings(df: DataFrame, rating_col: str = "rating", threshold: Optional[float] = None,
                     output_col: str = "label") -> DataFrame:
    """
    Convert ratings to binary values based on threshold.
    
    Args:
        df: Input Spark DataFrame
        rating_col: Column name containing ratings
        threshold: Threshold value (if None, mean rating is used)
        output_col: Column name for binary output
        
    Returns:
        DataFrame with binarized ratings
    """
    # Calculate mean if threshold not provided
    if threshold is None:
        threshold = df.groupBy().avg(rating_col).collect()[0][0]
        
    print(f"Using threshold: {threshold}")
    
    # Create UDF to binarize ratings
    binarize_udf = F.udf(lambda x: 0 if x < threshold else 1, 'int')
    result_df = df.withColumn(output_col, binarize_udf(F.col(rating_col)))
    
    if output_col != rating_col:
        result_df = result_df.drop(rating_col)
        
    return result_df


def split_dataset(df: DataFrame, ratios: List[float] = [0.7, 0.2, 0.1], 
                  seed: int = 42) -> List[DataFrame]:
    """
    Split a DataFrame into multiple parts.
    
    Args:
        df: Input Spark DataFrame
        ratios: List of ratios for splitting (should sum to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        List of DataFrames split according to ratios
    """
    if abs(sum(ratios) - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")
        
    return df.randomSplit(ratios, seed=seed)