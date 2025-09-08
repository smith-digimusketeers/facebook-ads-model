#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-Efficient ML Pipeline with Ensemble Learning
====================================================
Implements Voting, Stacking, and Weighted Averaging ensemble methods
with configurable batch size and sample size for quick development.
"""

import argparse
import os
import time
import warnings
import joblib
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterator
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings

# GPU imports with fallback
HAS_CUML = False
HAS_XGBOOST_GPU = False
HAS_LIGHTGBM_GPU = False

try:
    import cuml
    from cuml.linear_model import Ridge as cuRidge
    HAS_CUML = True
    print("✅ RAPIDS cuML available")
except ImportError:
    print("⚠️ RAPIDS cuML not available")

try:
    import xgboost as xgb
    test_model = xgb.XGBRegressor(device='cuda', n_estimators=1)
    test_model.fit([[1]], [1])
    HAS_XGBOOST_GPU = True
    print("✅ XGBoost GPU support available")
except:
    try:
        import xgboost as xgb
        print("✅ XGBoost available (CPU only)")
    except ImportError:
        print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    lgb.LGBMRegressor(device='gpu', n_estimators=1).fit([[1]], [1])
    HAS_LIGHTGBM_GPU = True
    print("✅ LightGBM GPU support available")
except:
    try:
        import lightgbm as lgb
        print("✅ LightGBM available (CPU only)")
    except ImportError:
        print("⚠️ LightGBM not available")

warnings.filterwarnings('ignore')


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()  # Call twice for good measure
    gc.collect()


def calculate_wape(y_true, y_pred):
    """Calculate Weighted Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    mask = denominator != 0
    smape = np.zeros_like(y_true)
    smape[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(smape)


def inverse_log1p(y_log):
    """Inverse of log1p transformation"""
    return np.expm1(y_log)


def evaluate_predictions(y_true_log, y_pred_log, target_name, dataset_type="VAL"):
    """Evaluate predictions in both log and original scale"""
    # Log scale metrics
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)
    
    # Convert to original scale
    y_true_orig = inverse_log1p(y_true_log)
    y_pred_orig = inverse_log1p(y_pred_log)
    
    # Original scale metrics
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    wape_orig = calculate_wape(y_true_orig, y_pred_orig)
    smape_orig = calculate_smape(y_true_orig, y_pred_orig)
    
    # Format output
    target_clean = target_name.replace('_log1p', '')
    
    print(f"[{dataset_type} LOG] MAE={mae_log:.4f} RMSE={rmse_log:.4f} R2={r2_log:.4f}")
    print(f"[{dataset_type} ORIG] {target_clean}: MAE={mae_orig:,.2f} RMSE={rmse_orig:,.2f} R2={r2_orig:.4f} WAPE={wape_orig:.3f} sMAPE={smape_orig:.3f}")
    
    return {
        'log': {
            'mae': mae_log,
            'rmse': rmse_log,
            'r2': r2_log
        },
        'orig': {
            'mae': mae_orig,
            'rmse': rmse_orig,
            'r2': r2_orig,
            'wape': wape_orig,
            'smape': smape_orig
        }
    }


class TrulyMemoryEfficientDataLoader:
    """Data loader that NEVER loads the full dataset"""
    
    def __init__(self, file_path: str, max_memory_mb: int = 8000, 
                 batch_size: Optional[int] = None, sample_size: Optional[int] = None):
        self.file_path = file_path
        self.max_memory_mb = max_memory_mb
        self.sample_size = sample_size  # New: limit total rows to use
        
        # Calculate batch size - use provided or calculate safe default
        if batch_size is not None:
            self.batch_size = batch_size
            print(f"📊 Using user-defined batch size: {self.batch_size:,}")
        else:
            self.batch_size = self._calculate_safe_batch_size()
            print(f"📊 Using auto-calculated batch size: {self.batch_size:,}")
        
        # Get basic info without loading data
        self._scan_data_info()
    
    def _scan_data_info(self):
        """Get data info without loading"""
        print("📊 Scanning dataset info...")
        lf = pl.scan_parquet(self.file_path)
        
        # Get total rows (this is fast - just metadata)
        self.total_rows = lf.select(pl.len()).collect().item()
        self.original_total_rows = self.total_rows
        
        # Apply sample size if specified
        if self.sample_size is not None and self.sample_size < self.total_rows:
            print(f"   🎯 Sampling {self.sample_size:,} rows from {self.total_rows:,} total rows")
            print(f"      ({self.sample_size/self.total_rows*100:.1f}% of data)")
            self.total_rows = self.sample_size
            self.is_sampled = True
        else:
            self.is_sampled = False
        
        # Get column names (also fast)
        self.columns = lf.collect_schema().names()
        
        # Get a tiny sample for column analysis
        tiny_sample = lf.head(1000).collect().to_pandas()
        self.dtypes = tiny_sample.dtypes
        
        print(f"   Total rows (original): {self.original_total_rows:,}")
        if self.is_sampled:
            print(f"   Total rows (sampled): {self.total_rows:,}")
        print(f"   Total columns: {len(self.columns)}")
        print(f"   Batch size: {self.batch_size:,}")
        
        del tiny_sample
        force_cleanup()
    
    def _calculate_safe_batch_size(self):
        """Calculate safe batch size based on available memory"""
        available_memory = self.max_memory_mb * 0.7  # Use 70% of limit
        # Estimate: ~1KB per row with 160 features
        estimated_row_size_kb = 1.0
        safe_batch_size = int((available_memory * 1024) / estimated_row_size_kb)
        return min(safe_batch_size, 50000)  # Cap at 50k rows
    
    def get_data_splits(self, target_cols: List[str], test_size: float = 0.15, 
                       val_size: float = 0.15, random_state: int = 42):
        """Get train/val/test splits without loading full data"""
        
        print(f"🔀 Creating data splits")
        
        # If using sample, randomly select indices
        if self.is_sampled:
            # Randomly select sample_size indices from the full dataset
            np.random.seed(random_state)
            all_indices = np.random.choice(
                self.original_total_rows, 
                size=self.sample_size, 
                replace=False
            )
            print(f"   Using {len(all_indices):,} sampled indices")
        else:
            # Use all indices
            all_indices = np.arange(self.total_rows)
        
        # Split indices (this is memory-free)
        train_idx, test_idx = train_test_split(
            all_indices, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"   Train indices: {len(train_idx):,}")
        print(f"   Val indices: {len(val_idx):,}")
        print(f"   Test indices: {len(test_idx):,}")
        
        return train_idx, val_idx, test_idx
    
    def load_data_subset(self, indices: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """Load only specified rows and columns"""
        print(f"📥 Loading {len(indices):,} rows, {len(columns)} columns")
        
        # Use the user-defined batch size for loading
        current_batch_size = self.batch_size
        
        if len(indices) > current_batch_size:
            print(f"   Processing in batches of {current_batch_size:,}")
        
        # Sort indices for better I/O performance
        sorted_indices = np.sort(indices)
        
        # Load data using polars with row selection
        lf = pl.scan_parquet(self.file_path)
        
        # Load in chunks based on batch_size
        chunks = []
        
        for i in range(0, len(sorted_indices), current_batch_size):
            chunk_indices = sorted_indices[i:i + current_batch_size]
            
            # Load specific rows using filter
            chunk_lf = lf.with_row_count("row_nr").filter(
                pl.col("row_nr").is_in(chunk_indices.tolist())
            ).drop("row_nr").select(columns)
            
            chunk_df = chunk_lf.collect().to_pandas()
            chunks.append(chunk_df)
            
            print(f"   Loaded batch {i//current_batch_size + 1}/{(len(sorted_indices)-1)//current_batch_size + 1}: "
                  f"{len(chunk_df)} rows, Memory: {get_memory_usage():.1f}MB")
            
            del chunk_lf
            force_cleanup()
        
        # Combine chunks
        if len(chunks) == 1:
            result_df = chunks[0]
        else:
            print("   Combining batches...")
            result_df = pd.concat(chunks, ignore_index=True)
            
            # Clean up chunks
            for chunk in chunks:
                del chunk
            del chunks
            force_cleanup()
        
        print(f"   Final subset shape: {result_df.shape}")
        print(f"   Memory after loading: {get_memory_usage():.1f}MB")
        
        return result_df


class EnsembleModel:
    """Wrapper for various ensemble methods"""
    
    def __init__(self, base_models: Dict[str, Any], ensemble_type: str = 'voting'):
        """
        Initialize ensemble model
        
        Args:
            base_models: Dictionary of trained base models
            ensemble_type: 'voting', 'weighted_voting', or 'stacking'
        """
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.weights = None
        self.meta_model = None
        
    def set_weights(self, weights: Dict[str, float]):
        """Set weights for weighted voting"""
        self.weights = weights
    
    def train_stacking_meta_model(self, X_val: pd.DataFrame, y_val: np.ndarray):
        """Train meta-model for stacking ensemble"""
        print("   Training stacking meta-model...")
        
        # Get predictions from all base models
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X_val)
            base_predictions.append(pred)
        
        # Stack predictions as features for meta-model
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-model (using Ridge for stability)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_meta, y_val)
        
        # Evaluate meta-model
        meta_pred = self.meta_model.predict(X_meta)
        meta_r2 = r2_score(y_val, meta_pred)
        print(f"   Meta-model R2 on validation: {meta_r2:.4f}")
        
        return meta_r2
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        
        # Get predictions from all base models
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        
        if self.ensemble_type == 'voting':
            # Simple average (equal weights)
            return np.mean(list(predictions.values()), axis=0)
        
        elif self.ensemble_type == 'weighted_voting':
            # Weighted average
            if self.weights is None:
                # Default to equal weights if not set
                return np.mean(list(predictions.values()), axis=0)
            
            weighted_sum = np.zeros_like(list(predictions.values())[0])
            total_weight = 0
            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                weighted_sum += weight * pred
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        elif self.ensemble_type == 'stacking':
            # Use meta-model to combine predictions
            if self.meta_model is None:
                raise ValueError("Meta-model not trained. Call train_stacking_meta_model first.")
            
            X_meta = np.column_stack(list(predictions.values()))
            return self.meta_model.predict(X_meta)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")


class BaggingEnsemble:
    """Bootstrap Aggregating ensemble for a single model type"""
    
    def __init__(self, base_estimator, n_estimators: int = 10, max_samples: float = 0.8):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train multiple models on bootstrap samples"""
        n_samples = len(X)
        sample_size = int(n_samples * self.max_samples)
        
        print(f"   Training {self.n_estimators} bagged models...")
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y[indices]
            
            # Clone and train model
            from sklearn.base import clone
            model = clone(self.base_estimator)
            model.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(model)
            
            if (i + 1) % 5 == 0:
                print(f"     Trained {i + 1}/{self.n_estimators} models")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average predictions from all models"""
        predictions = []
        for model in self.estimators:
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)


class EnsembleModelTrainer:
    """Enhanced trainer with ensemble learning capabilities"""
    
    def __init__(self, max_memory_mb: int = 8000, batch_size: Optional[int] = None):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size  # For incremental training
    
    def create_base_models(self, use_gpu: bool = True, random_state: int = 42, 
                          quick_mode: bool = False) -> Dict[str, Any]:
        """Create base models for ensemble"""
        models = {}
        
        # Reduce model complexity in quick mode
        n_estimators = 50 if quick_mode else 100
        max_depth = 4 if quick_mode else 6
        
        print(f"🔧 Creating models (quick_mode={quick_mode}, n_estimators={n_estimators})")
        
        # 1. SGD Ridge (incremental learning)
        models['sgd_ridge'] = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.01,
            random_state=random_state,
            max_iter=500 if quick_mode else 1000,
            tol=1e-3
        )
        
        # 2. XGBoost
        if HAS_XGBOOST_GPU and use_gpu:
            models['xgb'] = xgb.XGBRegressor(
                device='cuda',
                tree_method='hist',
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.15 if quick_mode else 0.1,
                random_state=random_state,
                verbosity=0,
                objective='reg:squarederror'
            )
        elif 'xgb' in globals():
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.15 if quick_mode else 0.1,
                random_state=random_state,
                verbosity=0,
                n_jobs=1,
                objective='reg:squarederror'
            )
        
        # 3. LightGBM
        if HAS_LIGHTGBM_GPU and use_gpu:
            models['lgb'] = lgb.LGBMRegressor(
                device='gpu',
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.15 if quick_mode else 0.1,
                random_state=random_state,
                verbosity=-1,
                objective='regression',
                num_leaves=15 if quick_mode else 31
            )
        elif 'lgb' in globals():
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.15 if quick_mode else 0.1,
                random_state=random_state,
                verbosity=-1,
                n_jobs=1,
                objective='regression',
                num_leaves=15 if quick_mode else 31
            )
        
        # 4. Ridge regression
        if HAS_CUML and use_gpu:
            models['ridge'] = cuRidge(alpha=1.0)
        else:
            models['ridge'] = Ridge(alpha=1.0, random_state=random_state)
        
        # 5. ElasticNet (skip in quick mode for speed)
        if not quick_mode:
            models['elastic_net'] = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=random_state,
                max_iter=1000
            )
        
        return models
    
    def train_single_model(self, model, model_name: str, X_train: pd.DataFrame, 
                          y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        """Train a single model with memory monitoring"""
        
        print(f"🔧 Training {model_name}...")
        start_time = time.time()
        initial_memory = get_memory_usage()
        
        try:
            # Check if model supports incremental learning
            if hasattr(model, 'partial_fit') and len(X_train) > 100000:
                print(f"   Using incremental training")
                
                # Use provided batch_size or calculate based on data size
                if self.batch_size is not None:
                    batch_size = min(self.batch_size, len(X_train))
                else:
                    batch_size = min(50000, len(X_train) // 4)
                    
                n_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size > 0 else 0)
                
                print(f"   Training in {n_batches} batches of {batch_size:,} rows")
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(X_train))
                    
                    X_batch = X_train.iloc[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
                    
                    if i == 0:
                        model.partial_fit(X_batch, y_batch)
                    else:
                        model.partial_fit(X_batch, y_batch)
                    
                    if i % 5 == 0:
                        current_memory = get_memory_usage()
                        print(f"     Batch {i+1}/{n_batches}, Memory: {current_memory:.1f}MB")
                        
                        if current_memory > self.max_memory_mb:
                            print("     Memory limit reached, forcing cleanup...")
                            force_cleanup()
            else:
                print(f"   Using standard training (data size: {len(X_train):,} rows)")
                model.fit(X_train, y_train)
            
            # Validation prediction
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            train_time = time.time() - start_time
            final_memory = get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            print(f"   ✅ {model_name}: Val R2={val_r2:.4f}, MAE={val_mae:.4f}, Time={train_time:.1f}s")
            
            return model, {
                'val_r2': val_r2,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'train_time': train_time,
                'memory_delta': memory_delta
            }
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {str(e)}")
            return None, None
        finally:
            force_cleanup()
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                            X_val: pd.DataFrame, y_val: np.ndarray,
                            base_models: Dict[str, Any],
                            ensemble_methods: List[str]) -> Tuple[Dict, Dict, Dict]:
        """Train all base models and create ensemble models"""
        
        # Train base models
        trained_models = {}
        model_results = {}
        
        for model_name, model in base_models.items():
            trained_model, results = self.train_single_model(
                model, model_name, X_train, y_train, X_val, y_val
            )
            
            if trained_model is not None:
                trained_models[model_name] = trained_model
                model_results[model_name] = results
        
        if len(trained_models) < 2:
            print("⚠️ Not enough models trained for ensemble")
            return trained_models, model_results, {}
        
        # Create ensemble models based on requested methods
        ensemble_results = {}
        
        if 'voting' in ensemble_methods:
            # 1. Simple Voting (Average)
            print("\n🔄 Creating Voting Ensemble...")
            voting_ensemble = EnsembleModel(trained_models, ensemble_type='voting')
            val_pred_voting = voting_ensemble.predict(X_val)
            voting_r2 = r2_score(y_val, val_pred_voting)
            voting_mae = mean_absolute_error(y_val, val_pred_voting)
            print(f"   Voting Ensemble: Val R2={voting_r2:.4f}, MAE={voting_mae:.4f}")
            
            ensemble_results['voting'] = {
                'model': voting_ensemble,
                'val_r2': voting_r2,
                'val_mae': voting_mae
            }
        
        if 'weighted_voting' in ensemble_methods:
            # 2. Weighted Voting
            print("\n🔄 Creating Weighted Voting Ensemble...")
            weights = {name: max(0, results['val_r2']) for name, results in model_results.items()}
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            weighted_ensemble = EnsembleModel(trained_models, ensemble_type='weighted_voting')
            weighted_ensemble.set_weights(normalized_weights)
            val_pred_weighted = weighted_ensemble.predict(X_val)
            weighted_r2 = r2_score(y_val, val_pred_weighted)
            weighted_mae = mean_absolute_error(y_val, val_pred_weighted)
            
            print(f"   Model weights: {', '.join([f'{k}: {v:.3f}' for k, v in normalized_weights.items()])}")
            print(f"   Weighted Voting: Val R2={weighted_r2:.4f}, MAE={weighted_mae:.4f}")
            
            ensemble_results['weighted_voting'] = {
                'model': weighted_ensemble,
                'val_r2': weighted_r2,
                'val_mae': weighted_mae,
                'weights': normalized_weights
            }
        
        if 'stacking' in ensemble_methods:
            # 3. Stacking Ensemble
            print("\n🔄 Creating Stacking Ensemble...")
            stacking_ensemble = EnsembleModel(trained_models, ensemble_type='stacking')
            stacking_r2 = stacking_ensemble.train_stacking_meta_model(X_val, y_val)
            val_pred_stacking = stacking_ensemble.predict(X_val)
            stacking_mae = mean_absolute_error(y_val, val_pred_stacking)
            
            print(f"   Stacking Ensemble: Val R2={stacking_r2:.4f}, MAE={stacking_mae:.4f}")
            
            ensemble_results['stacking'] = {
                'model': stacking_ensemble,
                'val_r2': stacking_r2,
                'val_mae': stacking_mae
            }
        
        if 'bagging' in ensemble_methods:
            # 4. Bagging (using best individual model)
            best_model_name = max(model_results.items(), key=lambda x: x[1]['val_r2'])[0]
            print(f"\n🔄 Creating Bagging Ensemble (using {best_model_name})...")
            
            if best_model_name in ['xgb', 'lgb']:
                print("   Skipping bagging for tree-based model (already uses internal bagging)")
            else:
                bagging_model = BaggingEnsemble(
                    base_estimator=base_models[best_model_name],
                    n_estimators=5,  # Keep low for memory efficiency
                    max_samples=0.8
                )
                bagging_model.fit(X_train, y_train)
                val_pred_bagging = bagging_model.predict(X_val)
                bagging_r2 = r2_score(y_val, val_pred_bagging)
                bagging_mae = mean_absolute_error(y_val, val_pred_bagging)
                
                print(f"   Bagging Ensemble: Val R2={bagging_r2:.4f}, MAE={bagging_mae:.4f}")
                
                ensemble_results['bagging'] = {
                    'model': bagging_model,
                    'val_r2': bagging_r2,
                    'val_mae': bagging_mae
                }
        
        return trained_models, model_results, ensemble_results


def main(args):
    print("🚀 Memory-Efficient ML Pipeline with Ensemble Learning")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Max Memory: {args.max_memory_mb}MB")
    print(f"Batch Size: {args.batch_size if args.batch_size else 'Auto'}")
    print(f"Sample Size: {args.sample_size if args.sample_size else 'All data'}")
    print(f"Ensemble Methods: {args.ensemble_methods}")
    print(f"Initial memory: {get_memory_usage():.1f}MB")
    
    # Determine if we're in quick development mode
    quick_mode = args.sample_size is not None and args.sample_size < 100000
    if quick_mode:
        print("⚡ Quick development mode enabled (small sample size)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader with batch_size and sample_size
    loader = TrulyMemoryEfficientDataLoader(
        args.input, 
        args.max_memory_mb,
        batch_size=args.batch_size,
        sample_size=args.sample_size
    )
    
    # Define columns
    target_cols = [
        'impressions_log1p', 'clicks_log1p', 'actions_log1p', 
        'reach_log1p', 'conversion_value_log1p'
    ]
    
    available_targets = [col for col in target_cols if col in loader.columns]
    feature_cols = [col for col in loader.columns if col not in target_cols]
    
    print(f"Available targets: {available_targets}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Get data splits (indices only - no memory used)
    train_idx, val_idx, test_idx = loader.get_data_splits(
        target_cols, args.test_size, args.val_size, args.random_state
    )
    
    # Initialize trainer with batch_size
    trainer = EnsembleModelTrainer(args.max_memory_mb, batch_size=args.batch_size)
    
    # Store all results
    all_results = {}
    best_models = {}
    best_models_info = {}  # For best_models_info.joblib
    
    for target in available_targets:
        target_clean = target.replace('_log1p', '')
        print(f"\n{'='*90}")
        print(f"\nTraining target: {target_clean}")
        
        # Load training data for this target
        print("📥 Loading training data...")
        train_cols = feature_cols + [target]
        train_data = loader.load_data_subset(train_idx, train_cols)
        train_data = train_data.dropna()
        print(f"   Training data after cleanup: {train_data.shape}")
        
        X_train = train_data[feature_cols]
        y_train = train_data[target].values
        del train_data
        force_cleanup()
        
        # Load validation data
        print("📥 Loading validation data...")
        val_data = loader.load_data_subset(val_idx, train_cols)
        val_data = val_data.dropna()
        
        X_val = val_data[feature_cols]
        y_val = val_data[target].values
        del val_data
        force_cleanup()
        
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"   Memory after data loading: {get_memory_usage():.1f}MB")
        
        # Create base models (with quick_mode if applicable)
        base_models = trainer.create_base_models(args.use_gpu, args.random_state, quick_mode)
        
        # Train models and create ensembles
        trained_models, model_results, ensemble_results = trainer.train_ensemble_models(
            X_train, y_train, X_val, y_val, base_models, args.ensemble_methods
        )
        
        # Find best model (including ensembles)
        all_models = {}
        all_models.update({k: v['val_r2'] for k, v in model_results.items()})
        all_models.update({f'ensemble_{k}': v['val_r2'] for k, v in ensemble_results.items()})
        
        if all_models:  # Check if we have any models
            best_model_name = max(all_models.items(), key=lambda x: x[1])[0]
            best_score = all_models[best_model_name]
            
            # Get best model object and handle ensemble vs individual models
            if best_model_name.startswith('ensemble_'):
                ensemble_type = best_model_name.replace('ensemble_', '')
                best_model = ensemble_results[ensemble_type]['model']
                model_info = ensemble_results[ensemble_type]
                
                # For ensemble models, we need to save differently
                is_ensemble = True
                ensemble_details = {
                    'ensemble_type': ensemble_type,
                    'base_models': list(trained_models.keys()),
                    'weights': model_info.get('weights', None)
                }
            else:
                best_model = trained_models[best_model_name]
                model_info = model_results[best_model_name]
                is_ensemble = False
                ensemble_details = None
            
            # Validation evaluation with formatted output
            val_pred = best_model.predict(X_val)
            val_metrics = evaluate_predictions(y_val, val_pred, target, "VAL")
            
            # Test evaluation
            print("📥 Loading test data...")
            test_data = loader.load_data_subset(test_idx, train_cols)
            test_data = test_data.dropna()
            
            X_test = test_data[feature_cols]
            y_test = test_data[target].values
            
            # Test predictions and evaluation
            test_pred = best_model.predict(X_test)
            test_metrics = evaluate_predictions(y_test, test_pred, target, "TEST")
            
            print("=" * 90)
            
            # Save models appropriately
            if is_ensemble:
                # For ensemble models, save base models individually and ensemble info separately
                model_path = os.path.join(args.output_dir, f'best_model_{target_clean}.joblib')
                
                # Save base models for this target
                base_model_paths = {}
                for name, base_model in trained_models.items():
                    base_path = os.path.join(args.output_dir, f'base_model_{target_clean}_{name}.joblib')
                    joblib.dump(base_model, base_path)
                    base_model_paths[name] = base_path
                
                # Create ensemble reconstruction info (no custom objects)
                ensemble_reconstruction = {
                    'ensemble_type': ensemble_type,
                    'base_model_paths': base_model_paths,
                    'base_model_names': list(trained_models.keys()),
                    'weights': ensemble_details.get('weights', None),
                    'meta_model_path': None  # Will be set if stacking
                }
                
                # Save meta-model separately for stacking
                if ensemble_type == 'stacking' and hasattr(best_model, 'meta_model'):
                    meta_path = os.path.join(args.output_dir, f'meta_model_{target_clean}.joblib')
                    joblib.dump(best_model.meta_model, meta_path)
                    ensemble_reconstruction['meta_model_path'] = meta_path
                
                # Save ensemble reconstruction info instead of the ensemble object
                joblib.dump(ensemble_reconstruction, model_path)
                print(f"   ✅ Saved ensemble reconstruction info and base models for {target_clean}")
                
            else:
                # For individual models, save normally
                model_path = os.path.join(args.output_dir, f'best_model_{target_clean}.joblib')
                joblib.dump(best_model, model_path)
                print(f"   ✅ Saved individual model for {target_clean}")
            
            # Save all models if requested
            if args.save_all_models:
                # Save individual models
                for name, model in trained_models.items():
                    model_path = os.path.join(args.output_dir, f'model_{target_clean}_{name}.joblib')
                    joblib.dump(model, model_path)
                
                # Save ensemble models (using reconstruction format)
                for name, ensemble_info in ensemble_results.items():
                    ensemble_path = os.path.join(args.output_dir, f'ensemble_{target_clean}_{name}.joblib')
                    
                    # Create ensemble reconstruction info instead of saving the object
                    ensemble_reconstruction = {
                        'ensemble_type': name,
                        'base_model_names': list(trained_models.keys()),
                        'weights': ensemble_info.get('weights', None),
                        'val_r2': ensemble_info.get('val_r2', 0.0),
                        'val_mae': ensemble_info.get('val_mae', 0.0),
                        'meta_model_data': None
                    }
                    
                    # Save meta-model separately for stacking
                    if name == 'stacking' and 'model' in ensemble_info:
                        ensemble_model = ensemble_info['model']
                        if hasattr(ensemble_model, 'meta_model'):
                            meta_path = os.path.join(args.output_dir, f'ensemble_{target_clean}_{name}_meta.joblib')
                            joblib.dump(ensemble_model.meta_model, meta_path)
                            ensemble_reconstruction['meta_model_path'] = meta_path
                    
                    joblib.dump(ensemble_reconstruction, ensemble_path)
                    print(f"   📁 Saved ensemble reconstruction: {name} for {target_clean}")
            
            # Store results for best_models_info.joblib (exclude model objects)
            clean_model_info = {
                'val_r2': model_info.get('val_r2', 0.0),
                'val_mae': model_info.get('val_mae', 0.0),
                'train_time': model_info.get('train_time', 0.0),
                'memory_delta': model_info.get('memory_delta', 0.0),
                'weights': model_info.get('weights', None) if is_ensemble else None,
                'ensemble_type': ensemble_type if is_ensemble else None
            }
            
            best_models_info[target] = {
                'model_name': best_model_name,
                'model_path': model_path,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_r2': val_metrics['log']['r2'],
                'test_r2': test_metrics['log']['r2'],
                'is_ensemble': is_ensemble,
                'ensemble_details': ensemble_details,
                'results': clean_model_info  # Clean version without model objects
            }
            
            # Store results for metadata (clean version without model objects)
            clean_ensemble_results = {}
            for k, v in ensemble_results.items():
                clean_ensemble_results[k] = {
                    'val_r2': v.get('val_r2', 0.0),
                    'val_mae': v.get('val_mae', 0.0),
                    'weights': v.get('weights', None),
                    'ensemble_type': k
                    # Explicitly exclude 'model' key
                }
            
            all_results[target] = {
                'model_results': model_results,
                'ensemble_results': clean_ensemble_results,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model': best_model_name,
                'best_score': best_score,
                'is_ensemble': is_ensemble
            }
            
            best_models[target] = {
                'model_name': best_model_name,
                'model_path': model_path,
                'val_r2': val_metrics['log']['r2'],
                'test_r2': test_metrics['log']['r2'],
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            del X_test, y_test, test_data
            force_cleanup()
        
        # Clean up
        del X_train, y_train, X_val, y_val
        if 'trained_models' in locals():
            del trained_models
        force_cleanup()
        
        print(f"\nMemory after target {target}: {get_memory_usage():.1f}MB")
    
    # Save best_models_info.joblib (compatible with original format)
    joblib.dump(best_models_info, os.path.join(args.output_dir, 'best_models_info.joblib'))
    
    # Save training_metadata.joblib with all details
    metadata = {
        'feature_columns': feature_cols,
        'target_columns': available_targets,
        'all_results': all_results,
        'best_models': best_models,
        'ensemble_methods_used': args.ensemble_methods,
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'quick_mode': quick_mode,
        'model_comparison': all_results  # For compatibility
    }
    
    joblib.dump(metadata, os.path.join(args.output_dir, 'training_metadata.joblib'))
    
    # Final summary
    print(f"\n{'='*90}")
    print(f"🎉 Training Complete!")
    print(f"{'='*90}")
    print(f"✅ Trained {len(best_models)} targets with ensemble methods")
    print(f"✅ Peak memory usage: {get_memory_usage():.1f}MB")
    
    if args.sample_size:
        print(f"✅ Used {args.sample_size:,} samples ({args.sample_size/loader.original_total_rows*100:.1f}% of data)")
    if args.batch_size:
        print(f"✅ Used batch size: {args.batch_size:,}")
    
    print(f"\nBest Models Summary:")
    print("-" * 90)
    print(f"{'Target':<25} {'Model':<20} {'Type':<10} {'Val R2':>8} {'Test R2':>8}")
    print("-" * 90)
    
    for target, info in best_models.items():
        target_clean = target.replace('_log1p', '')
        model_type = "Ensemble" if info['model_name'].startswith('ensemble_') else "Individual"
        print(f"{target_clean:<25} {info['model_name']:<20} {model_type:<10} "
              f"{info['val_r2']:>8.4f} {info['test_r2']:>8.4f}")
    
    # Performance comparison (original scale)
    print(f"\n📊 Model Performance (Original Scale):")
    print("-" * 90)
    print(f"{'Target':<25} {'Val WAPE':>10} {'Val sMAPE':>10} {'Test WAPE':>10} {'Test sMAPE':>10}")
    print("-" * 90)
    
    for target, info in best_models.items():
        target_clean = target.replace('_log1p', '')
        val_wape = info['val_metrics']['orig']['wape']
        val_smape = info['val_metrics']['orig']['smape']
        test_wape = info['test_metrics']['orig']['wape']
        test_smape = info['test_metrics']['orig']['smape']
        print(f"{target_clean:<25} {val_wape:>10.3f} {val_smape:>10.3f} "
              f"{test_wape:>10.3f} {test_smape:>10.3f}")
    
    print("\n✅ Saved 'best_models_info.joblib' with detailed metrics")
    print("✅ Saved 'training_metadata.joblib' with full training history")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Efficient ML Pipeline with Ensemble Learning")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--max-memory-mb", type=int, default=16000, help="Maximum memory usage in MB")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set size")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--ensemble-methods", nargs='+', 
                       default=['voting', 'weighted_voting', 'stacking'],
                       choices=['voting', 'weighted_voting', 'stacking', 'bagging'],
                       help="Ensemble methods to use")
    parser.add_argument("--save-all-models", action="store_true", 
                       help="Save all individual and ensemble models (not just best)")
    
    # New arguments for batch processing and sampling
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Number of rows to process in each batch (default: auto-calculate)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Total number of rows to use from dataset (for quick development)")
    
    args = parser.parse_args()
    main(args)