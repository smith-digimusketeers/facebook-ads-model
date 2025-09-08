#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-Accelerated Data Cleaning + Preprocessing for Facebook Ads Data (Streaming Final Output)
============================================================================================

Steps:
1. Remove duplicate rows (GPU-accelerated with Polars)
2. Outlier detection and removal using IsolationForest
3. One-hot encoding for categorical variables
4. Standardization for numeric variables (excluding OHE columns)
5. Memory-efficient batch processing for large datasets
6. Streaming final output for large datasets
7. GPU-first processing with CPU fallback

Output: Clean, preprocessed parquet file ready for ML pipeline
"""

import argparse
import os
import time
import warnings
import joblib
import gc
import psutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

# Target columns to exclude from standardization and outlier detection
Y_TARGETS = [
    'impressions_log1p',
    'clicks_log1p', 
    'actions_log1p',
    'reach_log1p',
    'conversion_value_log1p',
]

# Categorical columns for one-hot encoding
CATEGORICAL_COLS = [
    'campaign_objective_norm',
    'cta_type_norm',
    'impression_device_norm', 
    'business_type'
]


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def get_gpu_memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return float(result.stdout.strip())  # Already in MB
        return 0.0
    except:
        return 0.0


def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup including GPU memory."""
    # Force Python garbage collection
    gc.collect()
    
    # Try to clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    # Additional garbage collection
    gc.collect()
    
    # Small delay to allow cleanup
    time.sleep(0.1)


def check_memory_limits(current_memory_mb: float, gpu_memory_mb: float, 
                       ram_limit_mb: float = 45000, gpu_limit_mb: float = 12000) -> bool:
    """Check if memory usage is approaching limits."""
    ram_usage_pct = (current_memory_mb / ram_limit_mb) * 100
    gpu_usage_pct = (gpu_memory_mb / gpu_limit_mb) * 100
    
    if ram_usage_pct > 85 or gpu_usage_pct > 85:
        print(f"⚠️  HIGH MEMORY USAGE WARNING:")
        print(f"   RAM: {current_memory_mb:.1f} MB ({ram_usage_pct:.1f}% of {ram_limit_mb} MB)")
        print(f"   GPU: {gpu_memory_mb:.1f} MB ({gpu_usage_pct:.1f}% of {gpu_limit_mb} MB)")
        return True
    return False


def safe_collect(lf: pl.LazyFrame, prefer_gpu: bool = True) -> pl.DataFrame:
    """Collect with GPU if possible, else CPU fallback."""
    try:
        if prefer_gpu:
            return lf.collect(engine="gpu")
    except Exception as e:
        print(f"GPU collection failed: {e}, falling back to CPU")
        pass
    
    try:
        return lf.collect()
    except Exception as e:
        print(f"Collection error: {e}")
        raise


def get_data_info_gpu(input_path: str, prefer_gpu: bool = True) -> dict:
    """Get dataset info using GPU-accelerated Polars."""
    print(f"📊 Scanning data: {input_path}")
    
    lf = pl.scan_parquet(input_path)
    schema = lf.collect_schema()
    
    total_rows = safe_collect(lf.select(pl.len()), prefer_gpu).item()
    sample = safe_collect(lf.head(1000), prefer_gpu)
    estimated_memory_mb = (sample.estimated_size("mb") * total_rows) / 1000
    
    info = {
        'total_rows': total_rows,
        'columns': list(schema.names()),
        'schema': schema,
        'num_columns': len(schema),
        'estimated_memory_mb': estimated_memory_mb
    }
    
    print(f"Dataset info:")
    print(f"  - Rows: {total_rows:,}")
    print(f"  - Columns: {info['num_columns']}")
    print(f"  - Estimated memory: {estimated_memory_mb:.1f} MB")
    
    return info


def streaming_file_copy(source_path: str, dest_path: str, chunk_size: int = 200000, 
                       prefer_gpu: bool = True) -> None:
    """Copy large parquet file using streaming/batch processing."""
    print(f"\n📋 === Streaming File Copy (Memory-Efficient) ===")
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    
    # Get total rows for progress tracking
    lf = pl.scan_parquet(source_path)
    total_rows = safe_collect(lf.select(pl.len()), prefer_gpu).item()
    
    print(f"Total rows to copy: {total_rows:,}")
    
    # Method 1: Try lazy frame approach (most efficient)
    try:
        print("Attempting lazy frame copy (no memory materialization)...")
        lazy_df = pl.scan_parquet(source_path)
        
        # Use lazy operations to write directly without full materialization
        lazy_df.sink_parquet(dest_path, compression="zstd", statistics=True)
        
        print("✅ Lazy copy successful!")
        return
        
    except Exception as e:
        print(f"⚠️  Lazy copy failed: {e}")
        print("Falling back to chunked copy...")
    
    # Method 2: Chunked copy approach
    temp_chunk_files = []
    
    try:
        for i in range(0, total_rows, chunk_size):
            chunk_num = i // chunk_size + 1
            chunk_end = min(i + chunk_size, total_rows)
            
            print(f"Processing chunk {chunk_num}: rows {i:,} to {chunk_end:,}")
            
            # Load chunk
            chunk_lf = lf.slice(i, chunk_end - i)
            chunk_df = safe_collect(chunk_lf, prefer_gpu)
            
            # Save chunk temporarily
            temp_chunk_path = f"{dest_path}.temp_chunk_{chunk_num}.parquet"
            chunk_df.write_parquet(temp_chunk_path, compression="zstd")
            temp_chunk_files.append(temp_chunk_path)
            
            del chunk_df
            gc.collect()
            
            print(f"  → Saved chunk {chunk_num} ({len(temp_chunk_files)} total)")
        
        # Combine all chunks using lazy operations
        print(f"Combining {len(temp_chunk_files)} chunks into final file...")
        
        # Use lazy concat to combine without loading all into memory
        combined_lf = pl.concat([pl.scan_parquet(f) for f in temp_chunk_files], how="vertical_relaxed")
        combined_lf.sink_parquet(dest_path, compression="zstd", statistics=True)
        
        print("✅ Chunked copy successful!")
        
    except Exception as e:
        print(f"❌ Chunked copy failed: {e}")
        raise
        
    finally:
        # Cleanup temp files
        print("Cleaning up temporary files...")
        for temp_file in temp_chunk_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        if temp_chunk_files:
            print(f"Cleaned up {len(temp_chunk_files)} temporary files")


def remove_duplicate_rows_gpu(input_path: str, output_path: str, prefer_gpu: bool = True,
                             batch_size: int = 500000) -> int:
    """Remove duplicate rows using GPU-accelerated Polars."""
    print(f"\n🧹 === Removing Duplicate Rows (GPU-Accelerated) ===")
    
    lf = pl.scan_parquet(input_path)
    total_rows_before = safe_collect(lf.select(pl.len()), prefer_gpu).item()
    print(f"Total rows before deduplication: {total_rows_before:,}")
    
    data_info = get_data_info_gpu(input_path, prefer_gpu)
    
    if data_info['estimated_memory_mb'] > 10000:
        print(f"⚠️  Large dataset detected ({data_info['estimated_memory_mb']:.1f} MB)")
        print(f"Using batch processing with batch size: {batch_size:,}")
        unique_rows = remove_duplicates_batched_gpu(input_path, output_path, prefer_gpu, batch_size)
    else:
        print("Processing entire dataset at once...")
        lf_dedupe = lf.unique(maintain_order=True)
        df_clean = safe_collect(lf_dedupe, prefer_gpu)
        unique_rows = len(df_clean)
        df_clean.write_parquet(output_path, compression="zstd", statistics=True)
        
        # Clean up memory
        del df_clean
        aggressive_memory_cleanup()
    
    duplicates_removed = total_rows_before - unique_rows
    print(f"✅ Deduplication complete:")
    print(f"  Original: {total_rows_before:,} rows")
    print(f"  Unique: {unique_rows:,} rows") 
    print(f"  Removed: {duplicates_removed:,} duplicates ({duplicates_removed/total_rows_before:.1%})")
    
    return unique_rows


def remove_duplicates_batched_gpu(input_path: str, output_path: str, prefer_gpu: bool = True,
                                 batch_size: int = 500000) -> int:
    """Remove duplicates using batch processing for very large datasets."""
    lf = pl.scan_parquet(input_path)
    total_rows = safe_collect(lf.select(pl.len()), prefer_gpu).item()
    
    seen_hashes = set()
    temp_files = []
    unique_rows_total = 0
    
    print("Processing in batches...")
    for i in range(0, total_rows, batch_size):
        print(f"Batch {i//batch_size + 1}: processing rows {i:,} to {min(i+batch_size, total_rows):,}")
        
        batch_lf = lf.slice(i, min(batch_size, total_rows - i))
        batch_df = safe_collect(batch_lf, prefer_gpu)
        
        batch_with_hash = batch_df.with_columns(
            pl.concat_str([pl.col(c).cast(pl.Utf8) for c in batch_df.columns], separator="|")
            .hash()
            .alias("_row_hash")
        )
        
        before_filter = len(batch_with_hash)
        if seen_hashes:
            batch_unique = batch_with_hash.filter(~pl.col("_row_hash").is_in(list(seen_hashes)))
        else:
            batch_unique = batch_with_hash
        
        batch_unique = batch_unique.unique(subset=["_row_hash"], maintain_order=True)
        new_hashes = batch_unique.select("_row_hash").to_numpy().flatten()
        seen_hashes.update(new_hashes)
        
        batch_clean = batch_unique.drop("_row_hash")
        after_filter = len(batch_clean)
        
        print(f"  → {after_filter:,} unique rows ({before_filter - after_filter:,} duplicates)")
        
        if len(batch_clean) > 0:
            temp_path = f"{output_path}.batch_{i//batch_size}.parquet"
            batch_clean.write_parquet(temp_path, compression="zstd")
            temp_files.append(temp_path)
            unique_rows_total += len(batch_clean)
        
        del batch_df, batch_with_hash, batch_unique, batch_clean
        gc.collect()
    
    print(f"Combining {len(temp_files)} batches into final file...")
    if temp_files:
        lazy_frames = [pl.scan_parquet(f) for f in temp_files]
        combined_lf = pl.concat(lazy_frames, how="vertical_relaxed")
        combined_lf.sink_parquet(output_path, compression="zstd", statistics=True)
        
        for temp_file in temp_files:
            os.remove(temp_file)
        
        print(f"Final file saved: {output_path}")
        return unique_rows_total
    
    return 0


def train_outlier_detector_gpu(input_path: str, sample_size: int = 100000, 
                              contamination: float = 0.1, prefer_gpu: bool = True) -> tuple:
    """Train IsolationForest on sample using GPU-accelerated data loading."""
    print(f"\n🔍 === Training Outlier Detector ===")
    
    lf = pl.scan_parquet(input_path)
    sample_df = safe_collect(lf.head(sample_size), prefer_gpu)
    
    print(f"Sample size: {len(sample_df):,} rows")
    print(f"Sample columns: {sample_df.width}")
    
    sample_pd = sample_df.to_pandas()
    numeric_cols = sample_pd.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in Y_TARGETS]
    
    print(f"Using {len(numeric_features)} numeric features for outlier detection")
    print(f"Excluded target columns: {[col for col in Y_TARGETS if col in numeric_cols]}")
    
    X_sample = sample_pd[numeric_features].fillna(0)
    
    print(f"Training IsolationForest (contamination={contamination})...")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    iso_forest.fit(X_sample)
    training_time = time.time() - start_time
    
    print(f"✅ Outlier detector trained in {training_time:.1f}s")
    
    sample_predictions = iso_forest.predict(X_sample)
    outliers_in_sample = np.sum(sample_predictions == -1)
    print(f"Sample outlier detection: {outliers_in_sample:,} outliers ({outliers_in_sample/len(X_sample):.1%})")
    
    del sample_df, sample_pd, X_sample
    gc.collect()
    
    return iso_forest, numeric_features


def remove_outliers_gpu_fixed(input_path: str, output_path: str, iso_forest: IsolationForest,
                              numeric_features: list, prefer_gpu: bool = True, 
                              batch_size: int = 200000) -> int:
    """Remove outliers with schema consistency fixes."""
    print(f"\n🎯 === Removing Outliers (GPU-Accelerated Batching) ===")
    
    lf = pl.scan_parquet(input_path)
    total_rows = safe_collect(lf.select(pl.len()), prefer_gpu).item()
    original_schema = lf.collect_schema()
    print(f"Original schema: {len(original_schema)} columns")
    
    clean_rows_total = 0
    temp_files = []
    
    print(f"Processing {total_rows:,} rows in batches of {batch_size:,}")
    
    for i in range(0, total_rows, batch_size):
        batch_num = i // batch_size + 1
        batch_end = min(i + batch_size, total_rows)
        print(f"Batch {batch_num}: rows {i:,} to {batch_end:,}", end=' ')
        
        batch_lf = lf.slice(i, batch_end - i)
        batch_df = safe_collect(batch_lf, prefer_gpu)
        
        batch_pd = batch_df.to_pandas()
        available_features = [col for col in numeric_features if col in batch_pd.columns]
        X_batch = batch_pd[available_features].fillna(0)
        
        outlier_predictions = iso_forest.predict(X_batch)
        inlier_mask = outlier_predictions == 1
        inlier_series = pl.Series("inlier_mask", inlier_mask)
        batch_clean_df = batch_df.filter(inlier_series)
        
        outliers_removed = len(batch_df) - len(batch_clean_df)
        print(f"→ {len(batch_clean_df):,} clean ({outliers_removed:,} outliers)")
        
        if len(batch_clean_df) > 0:
            temp_path = f"{output_path}.clean_batch_{batch_num}.parquet"
            batch_clean_df.write_parquet(temp_path, compression="zstd")
            temp_files.append(temp_path)
            clean_rows_total += len(batch_clean_df)
        
        del batch_df, batch_pd, batch_clean_df, X_batch
        gc.collect()
    
    print(f"Combining {len(temp_files)} clean batches...")
    if temp_files:
        try:
            combined_lf = pl.concat([pl.scan_parquet(f) for f in temp_files], how="vertical_relaxed")
            combined_lf.sink_parquet(output_path, compression="zstd", statistics=True)
            
            print(f"✅ Outlier removal complete:")
            print(f"  Original: {total_rows:,} rows")
            print(f"  Clean: {clean_rows_total:,} rows")
            print(f"  Removed: {total_rows - clean_rows_total:,} outliers ({(total_rows - clean_rows_total)/total_rows:.1%})")
            
        except Exception as e:
            print(f"❌ Error combining batches: {e}")
            print("Attempting manual combination...")
            
            all_batches = []
            for temp_file in temp_files:
                batch_data = pl.read_parquet(temp_file)
                all_batches.append(batch_data)
            
            final_clean_df = pl.concat(all_batches, how="vertical_relaxed")
            final_clean_df.write_parquet(output_path, compression="zstd", statistics=True)
            print(f"✅ Manual combination successful: {len(final_clean_df):,} rows")
        
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return clean_rows_total
    
    return 0


def apply_one_hot_encoding_gpu(input_path: str, output_path: str, categorical_cols: List[str],
                              prefer_gpu: bool = True) -> Dict[str, List[str]]:
    """Apply one-hot encoding to categorical columns using GPU-accelerated Polars."""
    print(f"\n🔥 === One-Hot Encoding (GPU-Accelerated) ===")
    
    # Load data
    lf = pl.scan_parquet(input_path)
    df = safe_collect(lf, prefer_gpu)
    
    print(f"Input shape: {df.shape}")
    
    # Store original shape for cleanup
    original_shape = df.shape
    available_cats = [col for col in categorical_cols if col in df.columns]
    missing_cats = [col for col in categorical_cols if col not in df.columns]
    
    if missing_cats:
        print(f"⚠️  Missing categorical columns: {missing_cats}")
    if not available_cats:
        print("❌ No categorical columns found for encoding!")
        # Just copy file if no categorical columns
        df.write_parquet(output_path, compression="zstd", statistics=True)
        return {}
    
    print(f"Encoding {len(available_cats)} categorical columns: {available_cats}")
    
    # Store mapping of original column to one-hot columns
    ohe_mapping = {}
    
    for col in available_cats:
        print(f"  Encoding '{col}'...")
        
        # Get unique values (excluding nulls)
        unique_vals = df.select(pl.col(col)).drop_nulls().unique().to_series().to_list()
        unique_vals = sorted([str(val) for val in unique_vals])  # Ensure string and sort
        
        print(f"    {len(unique_vals)} unique values: {unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''}")
        
        # Create one-hot columns using Polars
        ohe_cols = []
        for val in unique_vals:
            ohe_col_name = f"{col}_{val}"
            # Create binary column: 1 if matches value, 0 otherwise
            df = df.with_columns(
                (pl.col(col).cast(pl.Utf8) == str(val)).cast(pl.Int8).alias(ohe_col_name)
            )
            ohe_cols.append(ohe_col_name)
        
        # Store mapping
        ohe_mapping[col] = ohe_cols
        
        # Drop original categorical column
        df = df.drop(col)
        
        print(f"    Created {len(ohe_cols)} binary columns, dropped original '{col}'")
    
    print(f"Final shape after OHE: {df.shape}")
    print(f"Added {sum(len(cols) for cols in ohe_mapping.values())} new binary columns")
    
    # Save result
    df.write_parquet(output_path, compression="zstd", statistics=True)
    
    print(f"✅ One-hot encoding complete")
    print(f"Final shape: {df.shape} (added {df.shape[1] - original_shape[1]} columns)")
    
    # Clean up memory
    del df
    aggressive_memory_cleanup()
    
    return ohe_mapping


def apply_standardization_gpu(input_path: str, output_path: str, exclude_cols: List[str],
                             ohe_mapping_path: str = None, prefer_gpu: bool = True) -> Dict[str, Tuple[float, float]]:
    """Apply standardization to numeric columns using GPU-accelerated Polars, excluding OHE columns."""
    print(f"\n📊 === Standardization (GPU-Accelerated) ===")
    
    # Load data
    lf = pl.scan_parquet(input_path)
    df = safe_collect(lf, prefer_gpu)
    
    print(f"Input shape: {df.shape}")
    
    # Store original shape for cleanup
    original_shape = df.shape
    
    # Load OHE mapping to exclude those columns from standardization
    ohe_columns = []
    if ohe_mapping_path and os.path.exists(ohe_mapping_path):
        print("Loading OHE mapping to exclude binary columns from standardization...")
        ohe_mapping = joblib.load(ohe_mapping_path)
        for original_col, encoded_cols in ohe_mapping.items():
            ohe_columns.extend(encoded_cols)
        print(f"Excluding {len(ohe_columns)} one-hot encoded columns from standardization")
    
    # Identify numeric columns using Polars schema
    schema = df.schema
    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }
    
    # Get all numeric columns
    all_numeric_cols = [col for col, dtype in schema.items() if dtype in numeric_dtypes]
    
    # Filter out excluded columns AND one-hot encoded columns
    numeric_cols_to_standardize = [col for col in all_numeric_cols 
                                  if col not in exclude_cols and col not in ohe_columns]
    
    print(f"Total numeric columns: {len(all_numeric_cols)}")
    print(f"One-hot encoded columns (excluded): {len(ohe_columns)}")
    print(f"Target columns (excluded): {len([col for col in exclude_cols if col in all_numeric_cols])}")
    print(f"Columns to standardize: {len(numeric_cols_to_standardize)}")
    
    if not numeric_cols_to_standardize:
        print("❌ No numeric columns to standardize!")
        df.write_parquet(output_path, compression="zstd", statistics=True)
        return {}
    
    # Show some examples of what we're standardizing vs not
    if ohe_columns:
        print(f"Examples of OHE columns (keeping as 0/1): {ohe_columns[:3]}...")
    if numeric_cols_to_standardize:
        print(f"Examples of columns being standardized: {numeric_cols_to_standardize[:3]}...")
    
    # Calculate mean and std for each column
    print("Computing means and standard deviations...")
    standardization_params = {}
    
    for col in numeric_cols_to_standardize:
        # Use Polars for GPU-accelerated statistics
        stats = df.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std")
        ])
        
        mean_val = stats.select("mean").item()
        std_val = stats.select("std").item()
        
        # Handle edge case where std is 0 or very small
        if std_val is None or std_val < 1e-8:
            print(f"    ⚠️  Column '{col}': std={std_val}, skipping standardization")
            continue
            
        standardization_params[col] = (mean_val, std_val)
        
        # Apply standardization: (x - mean) / std
        df = df.with_columns(
            ((pl.col(col) - mean_val) / std_val).alias(col)
        )
        
        print(f"    Standardized '{col}': mean={mean_val:.4f}, std={std_val:.4f}")
    
    print(f"✅ Standardized {len(standardization_params)} columns")
    print(f"✅ Kept {len(ohe_columns)} one-hot encoded columns as binary (0/1)")
    
    # Save result
    df.write_parquet(output_path, compression="zstd", statistics=True)
    
    print(f"Final shape: {df.shape}")
    print(f"Standardized {len(standardization_params)} columns, kept {len(ohe_columns)} OHE columns")
    
    # Clean up memory
    del df
    aggressive_memory_cleanup()
    
    return standardization_params


def main(args: argparse.Namespace) -> None:
    """Main data cleaning and preprocessing pipeline."""
    t0 = time.time()
    initial_memory = get_memory_usage_mb()
    initial_gpu_memory = get_gpu_memory_usage_mb()
    
    print(f"🚀 GPU-Accelerated Data Cleaning + Preprocessing Pipeline")
    print(f"Polars version: {pl.__version__}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Engine: {'GPU-preferred' if args.engine == 'gpu' else 'CPU-only'}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Final copy chunk size: {args.copy_chunk_size:,}")
    print(f"Initial memory usage: {initial_memory:.1f} MB RAM, {initial_gpu_memory:.1f} MB GPU")
    
    prefer_gpu = (args.engine == 'gpu')
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Step 1: Get data info
    data_info = get_data_info_gpu(args.input, prefer_gpu)
    current_memory = get_memory_usage_mb()
    current_gpu_memory = get_gpu_memory_usage_mb()
    print(f"After data info: {current_memory:.1f} MB RAM (+{current_memory - initial_memory:.1f}), {current_gpu_memory:.1f} MB GPU")
    
    # Check memory limits
    check_memory_limits(current_memory, current_gpu_memory)
    
    # Step 2: Remove duplicate rows
    print(f"\n{'='*60}")
    dedupe_path = args.output.replace('.parquet', '_deduplicated.parquet')
    
    if not os.path.exists(dedupe_path) or args.force_reprocess:
        unique_rows = remove_duplicate_rows_gpu(
            args.input, dedupe_path, prefer_gpu, args.batch_size
        )
        if unique_rows == 0:
            print("❌ No data after deduplication!")
            return
    else:
        print(f"Using existing deduplicated data: {dedupe_path}")
        unique_rows = safe_collect(pl.scan_parquet(dedupe_path).select(pl.len()), prefer_gpu).item()
        print(f"Existing deduplicated data: {unique_rows:,} rows")
    
    current_data_path = dedupe_path
    
    # Aggressive cleanup after deduplication
    aggressive_memory_cleanup()
    current_memory = get_memory_usage_mb()
    current_gpu_memory = get_gpu_memory_usage_mb()
    print(f"After deduplication cleanup: {current_memory:.1f} MB RAM, {current_gpu_memory:.1f} MB GPU")
    check_memory_limits(current_memory, current_gpu_memory)
    
    # Step 3: Outlier detection and removal
    if args.remove_outliers:
        print(f"\n{'='*60}")
        
        outlier_detector_path = args.output.replace('.parquet', '_outlier_detector.joblib')
        
        if not os.path.exists(outlier_detector_path) or args.force_reprocess:
            iso_forest, numeric_features = train_outlier_detector_gpu(
                current_data_path, args.outlier_sample_size, args.contamination, prefer_gpu
            )
            
            joblib.dump((iso_forest, numeric_features), outlier_detector_path)
            print(f"Outlier detector saved: {outlier_detector_path}")
        else:
            print(f"Loading existing outlier detector: {outlier_detector_path}")
            iso_forest, numeric_features = joblib.load(outlier_detector_path)
            print(f"Loaded outlier detector with {len(numeric_features)} features")
        
        clean_path = args.output.replace('.parquet', '_clean_no_outliers.parquet')
        
        if not os.path.exists(clean_path) or args.force_reprocess:
            clean_rows = remove_outliers_gpu_fixed(
                current_data_path, clean_path, iso_forest, numeric_features, 
                prefer_gpu, args.outlier_batch_size
            )
            if clean_rows == 0:
                print("❌ No data after outlier removal!")
                return
        else:
            print(f"Using existing outlier-free data: {clean_path}")
            clean_rows = safe_collect(pl.scan_parquet(clean_path).select(pl.len()), prefer_gpu).item()
            print(f"Existing clean data: {clean_rows:,} rows")
        
        current_data_path = clean_path
        
        # Aggressive cleanup after outlier removal
        aggressive_memory_cleanup()
        current_memory = get_memory_usage_mb()
        current_gpu_memory = get_gpu_memory_usage_mb()
        print(f"After outlier removal cleanup: {current_memory:.1f} MB RAM, {current_gpu_memory:.1f} MB GPU")
        check_memory_limits(current_memory, current_gpu_memory)
    
    # Step 4: One-Hot Encoding
    if args.apply_ohe:
        print(f"\n{'='*60}")
        ohe_path = args.output.replace('.parquet', '_one_hot_encoded.parquet')
        ohe_mapping_path = args.output.replace('.parquet', '_ohe_mapping.joblib')
        
        if not os.path.exists(ohe_path) or args.force_reprocess:
            ohe_mapping = apply_one_hot_encoding_gpu(
                current_data_path, ohe_path, CATEGORICAL_COLS, prefer_gpu
            )
            
            # Save OHE mapping for later use
            joblib.dump(ohe_mapping, ohe_mapping_path)
            print(f"OHE mapping saved: {ohe_mapping_path}")
        else:
            print(f"Using existing one-hot encoded data: {ohe_path}")
        
        current_data_path = ohe_path
        
        # Aggressive cleanup after OHE
        aggressive_memory_cleanup()
        current_memory = get_memory_usage_mb()
        current_gpu_memory = get_gpu_memory_usage_mb()
        print(f"After OHE cleanup: {current_memory:.1f} MB RAM, {current_gpu_memory:.1f} MB GPU")
        check_memory_limits(current_memory, current_gpu_memory)
    
    # Step 5: Standardization
    if args.apply_standardization:
        print(f"\n{'='*60}")
        standardized_path = args.output.replace('.parquet', '_standardized.parquet')
        standardization_params_path = args.output.replace('.parquet', '_standardization_params.joblib')
        
        if not os.path.exists(standardized_path) or args.force_reprocess:
            # Exclude target variables from standardization
            exclude_cols = Y_TARGETS.copy()
            
            # Pass OHE mapping path to exclude binary columns
            ohe_mapping_path = args.output.replace('.parquet', '_ohe_mapping.joblib')
            
            standardization_params = apply_standardization_gpu(
                current_data_path, standardized_path, exclude_cols, 
                ohe_mapping_path, prefer_gpu
            )
            
            # Save standardization parameters
            joblib.dump(standardization_params, standardization_params_path)
            print(f"Standardization parameters saved: {standardization_params_path}")
        else:
            print(f"Using existing standardized data: {standardized_path}")
        
        current_data_path = standardized_path
        
        # Aggressive cleanup after standardization
        aggressive_memory_cleanup()
        current_memory = get_memory_usage_mb()
        current_gpu_memory = get_gpu_memory_usage_mb()
        print(f"After standardization cleanup: {current_memory:.1f} MB RAM, {current_gpu_memory:.1f} MB GPU")
        check_memory_limits(current_memory, current_gpu_memory)
    
    # Step 6: Final output (STREAMING/BATCH APPROACH)
    if current_data_path != args.output:
        print(f"\n{'='*60}")
        
        # Check if we need streaming copy based on file size AND memory usage
        file_size_mb = os.path.getsize(current_data_path) / (1024**2)
        estimated_memory_needed = file_size_mb * 2  # Factor for processing overhead
        
        print(f"Final file size: {file_size_mb:.1f} MB")
        print(f"Estimated memory needed: {estimated_memory_needed:.1f} MB")
        print(f"Memory threshold: {args.memory_threshold_mb} MB")
        print(f"Current memory: {current_memory:.1f} MB RAM, {current_gpu_memory:.1f} MB GPU")
        
        # Force streaming if memory usage is high OR file is large
        force_streaming = (current_memory > 30000 or current_gpu_memory > 10000 or 
                          estimated_memory_needed > args.memory_threshold_mb)
        
        if force_streaming:
            print("🔄 Using streaming copy (high memory usage or large file)")
            streaming_file_copy(current_data_path, args.output, args.copy_chunk_size, prefer_gpu)
        else:
            print("📋 Using direct copy (sufficient memory)")
            try:
                final_lf = pl.scan_parquet(current_data_path)
                final_df = safe_collect(final_lf, prefer_gpu)
                final_df.write_parquet(args.output, compression="zstd", statistics=True)
                print(f"✅ Direct copy successful: {len(final_df):,} rows, {final_df.width} columns")
                
                # Clean up memory
                del final_df
                aggressive_memory_cleanup()
                
            except Exception as e:
                print(f"❌ Direct copy failed: {e}")
                print("🔄 Falling back to streaming copy...")
                streaming_file_copy(current_data_path, args.output, args.copy_chunk_size, prefer_gpu)
        
        print(f"✅ Final preprocessed data saved: {args.output}")
    
    # Final cleanup
    aggressive_memory_cleanup()
    final_memory = get_memory_usage_mb()
    final_gpu_memory = get_gpu_memory_usage_mb()
    
    # Summary
    total_time = time.time() - t0
    print(f"\n🎉 === Data Cleaning + Preprocessing Complete ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Original rows: {data_info['total_rows']:,}")
    
    final_rows = safe_collect(pl.scan_parquet(args.output).select(pl.len()), prefer_gpu).item()
    print(f"Final rows: {final_rows:,}")
    print(f"Total removed: {data_info['total_rows'] - final_rows:,} ({(data_info['total_rows'] - final_rows)/data_info['total_rows']:.1%})")
    print(f"Output file: {args.output}")
    print(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB RAM (Δ{final_memory - initial_memory:+.1f} MB)")
    print(f"GPU memory: {initial_gpu_memory:.1f} MB → {final_gpu_memory:.1f} MB GPU (Δ{final_gpu_memory - initial_gpu_memory:+.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU-Accelerated Data Cleaning + Preprocessing for FB Ads")
    
    # Input/Output
    ap.add_argument("--input", "-i", required=True, help="Input parquet file path")
    ap.add_argument("--output", "-o", required=True, help="Output clean parquet file path")
    
    # Processing options
    ap.add_argument("--engine", choices=["gpu", "cpu"], default="gpu", 
                    help="Processing engine preference (default: gpu)")
    ap.add_argument("--batch-size", type=int, default=500000, 
                    help="Batch size for deduplication (default: 500000)")
    ap.add_argument("--force-reprocess", action="store_true", 
                    help="Force reprocessing even if intermediate files exist")
    
    # Memory management options
    ap.add_argument("--copy-chunk-size", type=int, default=200000, 
                    help="Chunk size for final copy operation (default: 200000)")
    ap.add_argument("--memory-threshold-mb", type=int, default=8000, 
                    help="Memory threshold for streaming copy (default: 8000 MB)")
    
    # Outlier removal options
    ap.add_argument("--remove-outliers", action="store_true", 
                    help="Remove outliers using IsolationForest")
    ap.add_argument("--contamination", type=float, default=0.1, 
                    help="Contamination rate for IsolationForest (default: 0.1)")
    ap.add_argument("--outlier-sample-size", type=int, default=100000, 
                    help="Sample size for training outlier detector (default: 100000)")
    ap.add_argument("--outlier-batch-size", type=int, default=200000, 
                    help="Batch size for outlier removal (default: 200000)")
    
    # Preprocessing options
    ap.add_argument("--apply-ohe", action="store_true", 
                    help="Apply one-hot encoding to categorical variables")
    ap.add_argument("--apply-standardization", action="store_true", 
                    help="Apply standardization to numeric variables")
    
    args = ap.parse_args()
    main(args)