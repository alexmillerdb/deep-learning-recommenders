from pyspark.sql import DataFrame
from typing import Dict
from streaming.base.converters import dataframe_to_mds


def save_dataframe_to_mds(df: DataFrame, output_path: str, columns_spec: Dict[str, str], 
                          num_workers: int = 40, compression: str = 'zstd:7') -> None:
    """
    Save a DataFrame to Mosaic Dataset Streaming (MDS) format.
    
    Args:
        df: Input Spark DataFrame
        output_path: Output directory path
        columns_spec: Dictionary mapping column names to their types
        num_workers: Number of workers for parallel processing
        compression: Compression algorithm and level
    """
    print(f"Saving data to MDS format at: {output_path}")
    mds_kwargs = {
        'out': output_path, 
        'columns': columns_spec, 
        'compression': compression
    }
    
    dataframe_to_mds(
        df.repartition(num_workers), 
        merge_index=True, 
        mds_kwargs=mds_kwargs
    )