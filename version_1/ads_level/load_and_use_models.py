#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load and Use Trained Models Script
==================================
Analyzes and uses models from ensemble training output.
Loads best_models_info.joblib and training_metadata.joblib files.
Handles custom ensemble classes properly.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def check_file_integrity(file_path: Path):
    """Check if a joblib file is valid and not corrupted"""
    try:
        # Check file size
        file_size = file_path.stat().st_size
        print(f"   📁 File size: {file_size:,} bytes")
        
        if file_size == 0:
            print("   ❌ File is empty")
            return False
        
        # Try to read first few bytes to check format
        with open(file_path, 'rb') as f:
            first_bytes = f.read(10)
            print(f"   🔍 First bytes: {first_bytes}")
            
            # Check for joblib signature
            if first_bytes.startswith(b'\x80'):
                print("   ✅ Appears to be a pickle/joblib file")
                return True
            else:
                print("   ⚠️ File doesn't appear to be a valid pickle/joblib file")
                return False
                
    except Exception as e:
        print(f"   ❌ Error checking file integrity: {e}")
        return False


def load_model_files_safe(models_dir: str):
    """Load both joblib files from the models directory with error handling"""
    
    models_path = Path(models_dir)
    
    # Check if directory exists
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # List all files in the directory
    print(f"📁 Files in {models_dir}:")
    for file in models_path.iterdir():
        if file.is_file():
            print(f"   📄 {file.name} ({file.stat().st_size:,} bytes)")
    
    # Load best_models_info.joblib with custom class handling
    best_models_path = models_path / 'best_models_info.joblib'
    if not best_models_path.exists():
        raise FileNotFoundError(f"best_models_info.joblib not found in {models_dir}")
    
    print(f"\n📂 Loading best_models_info.joblib from {models_dir}...")
    
    # Check file integrity first
    if not check_file_integrity(best_models_path):
        print("❌ File appears to be corrupted or invalid")
        return None, None
    
    try:
        # Try to load normally first
        best_models_info = joblib.load(best_models_path)
        print("✅ Successfully loaded best_models_info.joblib")
    except Exception as e:
        print(f"⚠️ Error loading best_models_info.joblib: {e}")
        print("🔧 Attempting to load with custom class handling...")
        
        # Try to load with custom class handling
        try:
            # Create a custom unpickler that can handle missing classes
            import pickle
            
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle EnsembleModel class
                    if name == 'EnsembleModel':
                        return create_dummy_ensemble_model()
                    # Handle BaggingEnsemble class
                    elif name == 'BaggingEnsemble':
                        return create_dummy_bagging_ensemble()
                    return super().find_class(module, name)
            
            with open(best_models_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                best_models_info = unpickler.load()
            
            print("✅ Successfully loaded with custom class handling")
            
        except Exception as e2:
            print(f"❌ Failed to load best_models_info.joblib: {e2}")
            print("💡 The file appears to be corrupted. Try:")
            print("   1. Re-running the training script")
            print("   2. Checking if the training completed successfully")
            print("   3. Verifying disk space during training")
            
            # Try to create a minimal fallback
            print("🔧 Creating minimal fallback data structure...")
            best_models_info = None  # Will be created after loading training_metadata
    
    # Load training_metadata.joblib
    metadata_path = models_path / 'training_metadata.joblib'
    if not metadata_path.exists():
        print(f"⚠️ training_metadata.joblib not found in {models_dir}")
        print("🔧 Creating minimal fallback metadata...")
        training_metadata = create_fallback_metadata()
    else:
        print(f"\n📂 Loading training_metadata.joblib from {models_dir}...")
        
        # Check file integrity
        if not check_file_integrity(metadata_path):
            print("❌ Metadata file appears to be corrupted")
            training_metadata = create_fallback_metadata()
        else:
            try:
                training_metadata = joblib.load(metadata_path)
                print("✅ Successfully loaded training_metadata.joblib")
            except Exception as e:
                print(f"❌ Error loading training_metadata.joblib: {e}")
                print("🔧 Creating fallback metadata...")
                training_metadata = create_fallback_metadata()
    
    # If best_models_info is None (corrupted), reconstruct from training_metadata
    if best_models_info is None and training_metadata:
        print("🔧 Reconstructing best_models_info from training_metadata...")
        best_models_info = create_fallback_models_info(training_metadata, models_dir)
    
    return best_models_info, training_metadata


def create_fallback_models_info(training_metadata=None, models_dir="."):
    """Create a fallback structure when best_models_info.joblib is corrupted"""
    print("🔧 Creating fallback models info structure...")
    
    models_path = Path(models_dir)
    fallback_info = {}
    
    # Use training_metadata if available to get accurate info
    if training_metadata and 'best_models' in training_metadata:
        print("   📊 Using data from training_metadata to reconstruct best_models_info...")
        
        for target, model_info in training_metadata['best_models'].items():
            # Construct model path
            model_path = model_info.get('model_path', f'./models_memory_efficient/best_model_{target.replace("_log1p", "")}.joblib')
            
            fallback_info[target] = {
                'model_name': model_info.get('model_name', 'unknown'),
                'model_path': model_path,
                'val_metrics': model_info.get('val_metrics', {
                    'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                    'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                }),
                'test_metrics': model_info.get('test_metrics', {
                    'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                    'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                })
            }
            print(f"   ✅ Reconstructed info for {target.replace('_log1p', '')}: {model_info.get('model_name', 'unknown')}")
    
    else:
        print("   🔍 Searching for individual model files...")
        
        # Common target names
        targets = ['impressions_log1p', 'clicks_log1p', 'actions_log1p', 'reach_log1p', 'conversion_value_log1p']
        
        for target in targets:
            target_clean = target.replace('_log1p', '')
            
            # Look for model files
            model_files = list(models_path.glob(f"*{target_clean}*"))
            if model_files:
                model_file = model_files[0]  # Take first match
                print(f"   📄 Found model file for {target_clean}: {model_file.name}")
                
                fallback_info[target] = {
                    'model_name': 'unknown',
                    'model_path': str(model_file),
                    'val_metrics': {
                        'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                        'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                    },
                    'test_metrics': {
                        'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                        'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                    }
                }
            else:
                print(f"   ❌ No model file found for {target_clean}")
        
        if not fallback_info:
            print("⚠️ No model files found. Creating empty structure...")
            # Create empty structure for common targets
            for target in targets:
                fallback_info[target] = {
                    'model_name': 'not_found',
                    'model_path': 'not_found',
                    'val_metrics': {
                        'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                        'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                    },
                    'test_metrics': {
                        'log': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                        'orig': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'wape': 0.0, 'smape': 0.0}
                    }
                }
    
    return fallback_info


def create_fallback_metadata():
    """Create minimal fallback metadata when files are corrupted"""
    print("🔧 Creating fallback metadata structure...")
    
    return {
        'feature_columns': [],
        'target_columns': ['impressions_log1p', 'clicks_log1p', 'actions_log1p', 'reach_log1p', 'conversion_value_log1p'],
        'ensemble_methods_used': [],
        'batch_size': None,
        'sample_size': None,
        'quick_mode': False,
        'all_results': {}
    }


def create_dummy_ensemble_model():
    """Create a dummy EnsembleModel class for loading purposes"""
    
    class DummyEnsembleModel:
        def __init__(self, *args, **kwargs):
            self.base_models = {}
            self.ensemble_type = 'unknown'
            self.weights = None
            self.meta_model = None
        
        def predict(self, X):
            # Return dummy predictions
            return np.zeros(len(X))
        
        def set_weights(self, weights):
            self.weights = weights
        
        def train_stacking_meta_model(self, X_val, y_val):
            return 0.0
    
    return DummyEnsembleModel


def create_dummy_bagging_ensemble():
    """Create a dummy BaggingEnsemble class for loading purposes"""
    
    class DummyBaggingEnsemble:
        def __init__(self, *args, **kwargs):
            self.base_estimator = None
            self.n_estimators = 0
            self.max_samples = 0.8
            self.estimators = []
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            # Return dummy predictions
            return np.zeros(len(X))
    
    return DummyBaggingEnsemble


def load_individual_models(best_models_info: Dict, models_dir: str):
    """Load individual models that can be used for prediction"""
    
    print(f"\n🔧 LOADING INDIVIDUAL MODELS FOR PREDICTION")
    print("=" * 60)
    
    loaded_models = {}
    models_path = Path(models_dir)
    
    for target, model_info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        model_path = model_info['model_path']
        
        # Check if it's a relative path and make it absolute
        if not Path(model_path).is_absolute():
            model_path = models_path / model_path
        
        print(f"🎯 Loading {target_clean} model from {model_path}...")
        
        try:
            # Try to load the model
            model = joblib.load(model_path)
            loaded_models[target] = {
                'model': model,
                'model_name': model_info['model_name'],
                'target_clean': target_clean
            }
            print(f"   ✅ Successfully loaded {model_info['model_name']}")
            
        except Exception as e:
            print(f"   ❌ Error loading {target_clean}: {e}")
            print(f"   📁 Model path: {model_path}")
            print(f"   📁 Path exists: {Path(model_path).exists()}")
            
            # Try to find the model file
            if not Path(model_path).exists():
                print(f"   🔍 Searching for model files...")
                model_files = list(models_path.glob(f"*{target_clean}*"))
                if model_files:
                    print(f"   📄 Found potential model files: {[f.name for f in model_files]}")
                else:
                    print(f"   ❌ No model files found for {target_clean}")
    
    print(f"\n✅ Successfully loaded {len(loaded_models)} models for prediction")
    return loaded_models


def analyze_best_models_info(best_models_info: Dict):
    """Analyze contents of best_models_info.joblib"""
    
    print("\n🔍 BEST_MODELS_INFO.JOBLIB ANALYSIS")
    print("=" * 80)
    
    print("This file contains the BEST model for each target with:")
    print("• Model name and file path")
    print("• Validation metrics (log and original scale)")
    print("• Test metrics (log and original scale)")
    print("• Model performance details")
    
    print(f"\n📊 Found {len(best_models_info)} trained targets:")
    
    # Summary table
    summary_data = []
    for target, info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        
        # Safely extract metrics
        try:
            val_r2 = info['val_metrics']['log']['r2']
            test_r2 = info['test_metrics']['log']['r2']
            model_name = info['model_name']
        except KeyError as e:
            print(f"⚠️ Missing metric for {target_clean}: {e}")
            val_r2 = 0.0
            test_r2 = 0.0
            model_name = "Unknown"
        
        summary_data.append({
            'Target': target_clean,
            'Best Model': model_name,
            'Val R²': val_r2,
            'Test R²': test_r2,
            'Model Path': info.get('model_path', 'Unknown')
        })
        
        print(f"  {target_clean:20} → {model_name:20} (Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f})")
    
    # Detailed metrics for each target
    print(f"\n📈 DETAILED METRICS BY TARGET:")
    print("=" * 80)
    
    for target, info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        print(f"\n🎯 {target_clean.upper()}")
        print("-" * 40)
        
        try:
            # Validation metrics
            val_log = info['val_metrics']['log']
            val_orig = info['val_metrics']['orig']
            print(f"[VAL LOG]  MAE={val_log['mae']:.4f} RMSE={val_log['rmse']:.4f} R²={val_log['r2']:.4f}")
            print(f"[VAL ORIG] MAE={val_orig['mae']:,.2f} RMSE={val_orig['rmse']:,.2f} R²={val_orig['r2']:.4f} WAPE={val_orig['wape']:.3f} sMAPE={val_orig['smape']:.3f}")
            
            # Test metrics
            test_log = info['test_metrics']['log']
            test_orig = info['test_metrics']['orig']
            print(f"[TEST LOG]  MAE={test_log['mae']:.4f} RMSE={test_log['rmse']:.4f} R²={test_log['r2']:.4f}")
            print(f"[TEST ORIG] MAE={test_orig['mae']:,.2f} RMSE={test_orig['rmse']:,.2f} R²={test_orig['r2']:.4f} WAPE={test_orig['wape']:.3f} sMAPE={test_orig['smape']:.3f}")
            
        except KeyError as e:
            print(f"⚠️ Missing metrics for {target_clean}: {e}")
        
        # Model info
        print(f"Model: {info.get('model_name', 'Unknown')}")
        print(f"Path: {info.get('model_path', 'Unknown')}")
    
    return pd.DataFrame(summary_data)


def analyze_training_metadata(training_metadata: Dict):
    """Analyze contents of training_metadata.joblib"""
    
    print("\n🔍 TRAINING_METADATA.JOBLIB ANALYSIS")
    print("=" * 80)
    
    print("This file contains COMPLETE training history:")
    print("• Feature and target column lists")
    print("• Performance of ALL models (individual + ensemble)")
    print("• Ensemble method comparisons")
    print("• Training configuration details")
    print("• Model comparison results")
    
    # Basic info
    print(f"\n📋 TRAINING CONFIGURATION:")
    print(f"  Feature columns: {len(training_metadata.get('feature_columns', []))}")
    print(f"  Target columns: {len(training_metadata.get('target_columns', []))}")
    print(f"  Ensemble methods: {training_metadata.get('ensemble_methods_used', 'Unknown')}")
    print(f"  Batch size: {training_metadata.get('batch_size', 'Auto')}")
    print(f"  Sample size: {training_metadata.get('sample_size', 'Full dataset')}")
    print(f"  Quick mode: {training_metadata.get('quick_mode', False)}")
    
    # Model comparison analysis
    print(f"\n📊 MODEL COMPARISON ANALYSIS:")
    print("=" * 80)
    
    comparison_data = []
    all_results = training_metadata.get('all_results', {})
    
    for target, results in all_results.items():
        target_clean = target.replace('_log1p', '')
        
        # Individual models
        model_results = results.get('model_results', {})
        for model_name, metrics in model_results.items():
            comparison_data.append({
                'Target': target_clean,
                'Model': model_name,
                'Type': 'Individual',
                'Val R²': metrics.get('val_r2', 0),
                'Val MAE': metrics.get('val_mae', 0),
                'Train Time (s)': metrics.get('train_time', 0)
            })
        
        # Ensemble models
        ensemble_results = results.get('ensemble_results', {})
        for model_name, metrics in ensemble_results.items():
            comparison_data.append({
                'Target': target_clean,
                'Model': f"ensemble_{model_name}",
                'Type': 'Ensemble',
                'Val R²': metrics.get('val_r2', 0),
                'Val MAE': metrics.get('val_mae', 0),
                'Train Time (s)': 0
            })
    
    if not comparison_data:
        print("⚠️ No model comparison data found")
        return pd.DataFrame(), pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Show best model per target
    print("\n🏆 BEST MODEL PER TARGET:")
    print("-" * 60)
    for target in comparison_df['Target'].unique():
        target_df = comparison_df[comparison_df['Target'] == target]
        if len(target_df) > 0:
            best_row = target_df.loc[target_df['Val R²'].idxmax()]
            print(f"{target:20} → {best_row['Model']:20} ({best_row['Type']:10}) R²: {best_row['Val R²']:.4f}")
    
    # Ensemble vs Individual comparison
    print(f"\n📈 ENSEMBLE VS INDIVIDUAL PERFORMANCE:")
    print("-" * 60)
    
    ensemble_analysis = []
    for target in comparison_df['Target'].unique():
        target_df = comparison_df[comparison_df['Target'] == target]
        
        # Best individual
        individual_df = target_df[target_df['Type'] == 'Individual']
        if len(individual_df) > 0:
            best_individual = individual_df['Val R²'].max()
            best_individual_name = individual_df.loc[individual_df['Val R²'].idxmax(), 'Model']
        else:
            best_individual = 0
            best_individual_name = "None"
        
        # Best ensemble
        ensemble_df = target_df[target_df['Type'] == 'Ensemble']
        if len(ensemble_df) > 0:
            best_ensemble = ensemble_df['Val R²'].max()
            best_ensemble_name = ensemble_df.loc[ensemble_df['Val R²'].idxmax(), 'Model']
            
            # Calculate improvement
            improvement = (best_ensemble - best_individual) / abs(best_individual) * 100 if best_individual != 0 else 0
            sign = "+" if improvement > 0 else ""
            
            ensemble_analysis.append({
                'Target': target,
                'Best Individual': best_individual,
                'Best Individual Name': best_individual_name,
                'Best Ensemble': best_ensemble,
                'Best Ensemble Name': best_ensemble_name,
                'Improvement (%)': improvement
            })
            
            print(f"{target:15} → Individual: {best_individual:.4f} ({best_individual_name})")
            print(f"{'':<15}    Ensemble:   {best_ensemble:.4f} ({best_ensemble_name}) [{sign}{improvement:.1f}%]")
        else:
            print(f"{target:15} → Individual: {best_individual:.4f} (No ensemble trained)")
    
    return comparison_df, pd.DataFrame(ensemble_analysis)


def create_visualizations(best_models_info: Dict, training_metadata: Dict, output_dir: str):
    """Create visualizations of model performance"""
    
    print(f"\n📊 CREATING VISUALIZATIONS in {output_dir}")
    print("=" * 60)
    
    # Create output directory for plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Performance comparison across targets
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Across Targets', fontsize=16)
        
        targets = list(best_models_info.keys())
        target_names = [t.replace('_log1p', '') for t in targets]
        
        # R² scores
        val_r2_scores = []
        test_r2_scores = []
        
        for t in targets:
            try:
                val_r2_scores.append(best_models_info[t]['val_metrics']['log']['r2'])
                test_r2_scores.append(best_models_info[t]['test_metrics']['log']['r2'])
            except KeyError:
                val_r2_scores.append(0)
                test_r2_scores.append(0)
        
        x = np.arange(len(target_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, val_r2_scores, width, label='Validation R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        axes[0, 0].set_xlabel('Targets')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(target_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # WAPE scores
        val_wape_scores = []
        test_wape_scores = []
        
        for t in targets:
            try:
                val_wape_scores.append(best_models_info[t]['val_metrics']['orig']['wape'])
                test_wape_scores.append(best_models_info[t]['test_metrics']['orig']['wape'])
            except KeyError:
                val_wape_scores.append(0)
                test_wape_scores.append(0)
        
        axes[0, 1].bar(x - width/2, val_wape_scores, width, label='Validation WAPE', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_wape_scores, width, label='Test WAPE', alpha=0.8)
        axes[0, 1].set_xlabel('Targets')
        axes[0, 1].set_ylabel('WAPE Score')
        axes[0, 1].set_title('WAPE Performance Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(target_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model types used
        model_types = []
        for t in targets:
            try:
                model_types.append(best_models_info[t]['model_name'])
            except KeyError:
                model_types.append('Unknown')
        
        model_type_counts = pd.Series(model_types).value_counts()
        
        axes[1, 0].pie(model_type_counts.values, labels=model_type_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Distribution of Best Model Types')
        
        # Performance heatmap
        metrics = ['val_r2', 'test_r2', 'val_wape', 'test_wape']
        metric_names = ['Val R²', 'Test R²', 'Val WAPE', 'Test WAPE']
        
        heatmap_data = []
        for target in targets:
            row = []
            try:
                row.append(best_models_info[target]['val_metrics']['log']['r2'])
                row.append(best_models_info[target]['test_metrics']['log']['r2'])
                row.append(best_models_info[target]['val_metrics']['orig']['wape'])
                row.append(best_models_info[target]['test_metrics']['orig']['wape'])
            except KeyError:
                row = [0, 0, 0, 0]
            heatmap_data.append(row)
        
        im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels(metric_names, rotation=45)
        axes[1, 1].set_yticks(range(len(target_names)))
        axes[1, 1].set_yticklabels(target_names)
        axes[1, 1].set_title('Performance Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Performance Score')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved performance overview to {plots_dir}/performance_overview.png")
        
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")
    
    return plots_dir


def predict_with_models(loaded_models: Dict, sample_data: pd.DataFrame, output_dir: str):
    """Make predictions using loaded models"""
    
    print(f"\n🔮 MAKING PREDICTIONS WITH LOADED MODELS")
    print("=" * 60)
    
    predictions = {}
    prediction_summary = []
    
    for target, model_info in loaded_models.items():
        target_clean = model_info['target_clean']
        model = model_info['model']
        model_name = model_info['model_name']
        
        print(f"🎯 Predicting {target_clean} using {model_name}...")
        
        try:
            # Make predictions
            pred_log = model.predict(sample_data)
            pred_orig = np.expm1(pred_log)  # Convert to original scale
            
            predictions[target_clean] = {
                'predictions_log': pred_log,
                'predictions_original': pred_orig,
                'model_name': model_name
            }
            
            # Summary stats
            summary = {
                'Target': target_clean,
                'Model': model_name,
                'Mean Prediction': pred_orig.mean(),
                'Std Prediction': pred_orig.std(),
                'Min Prediction': pred_orig.min(),
                'Max Prediction': pred_orig.max(),
                'Sample Size': len(pred_orig)
            }
            prediction_summary.append(summary)
            
            print(f"   ✅ {target_clean}: Mean = {pred_orig.mean():,.0f}, Range = {pred_orig.min():,.0f} to {pred_orig.max():,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error predicting {target_clean}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(prediction_summary)
    
    # Save predictions
    predictions_file = Path(output_dir) / "model_predictions.csv"
    results_df.to_csv(predictions_file, index=False)
    
    print(f"\n✅ Saved prediction summary to {predictions_file}")
    
    # Create detailed predictions file
    detailed_predictions = {}
    for target, preds in predictions.items():
        detailed_predictions[f"{target}_log"] = preds['predictions_log']
        detailed_predictions[target] = preds['predictions_original']
    
    detailed_df = pd.DataFrame(detailed_predictions)
    detailed_file = Path(output_dir) / "detailed_predictions.csv"
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"✅ Saved detailed predictions to {detailed_file}")
    
    return predictions, results_df


def export_analysis_results(best_models_info: Dict, training_metadata: Dict, output_dir: str):
    """Export comprehensive analysis results"""
    
    print(f"\n💾 EXPORTING ANALYSIS RESULTS to {output_dir}")
    print("=" * 60)
    
    # 1. Best models summary
    best_models_summary = analyze_best_models_info(best_models_info)
    best_models_file = Path(output_dir) / "best_models_summary.csv"
    best_models_summary.to_csv(best_models_file, index=False)
    
    # 2. Complete model comparison
    comparison_df, ensemble_analysis = analyze_training_metadata(training_metadata)
    comparison_file = Path(output_dir) / "complete_model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    # 3. Ensemble analysis
    if len(ensemble_analysis) > 0:
        ensemble_file = Path(output_dir) / "ensemble_analysis.csv"
        ensemble_analysis.to_csv(ensemble_file, index=False)
    
    # 4. Performance metrics by target
    performance_data = []
    for target, info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        
        try:
            val_metrics = info['val_metrics']
            test_metrics = info['test_metrics']
            
            performance_data.append({
                'Target': target_clean,
                'Best_Model': info['model_name'],
                'Val_R2_Log': val_metrics['log']['r2'],
                'Test_R2_Log': test_metrics['log']['r2'],
                'Val_R2_Orig': val_metrics['orig']['r2'],
                'Test_R2_Orig': test_metrics['orig']['r2'],
                'Val_WAPE': val_metrics['orig']['wape'],
                'Test_WAPE': test_metrics['orig']['wape'],
                'Val_sMAPE': val_metrics['orig']['smape'],
                'Test_sMAPE': test_metrics['orig']['smape']
            })
        except KeyError as e:
            print(f"⚠️ Missing performance data for {target_clean}: {e}")
    
    performance_df = pd.DataFrame(performance_data)
    performance_file = Path(output_dir) / "performance_metrics.csv"
    performance_df.to_csv(performance_file, index=False)
    
    # 5. Feature information
    feature_info = {
        'feature_columns': training_metadata.get('feature_columns', []),
        'target_columns': training_metadata.get('target_columns', []),
        'total_features': len(training_metadata.get('feature_columns', [])),
        'total_targets': len(training_metadata.get('target_columns', [])),
        'ensemble_methods': training_metadata.get('ensemble_methods_used', []),
        'training_config': {
            'batch_size': training_metadata.get('batch_size'),
            'sample_size': training_metadata.get('sample_size'),
            'quick_mode': training_metadata.get('quick_mode', False)
        }
    }
    
    feature_file = Path(output_dir) / "feature_info.joblib"
    joblib.dump(feature_info, feature_file)
    
    print("✅ Exported files:")
    print(f"   📄 {best_models_file}")
    print(f"   📄 {comparison_file}")
    print(f"   📄 {performance_file}")
    print(f"   📄 {feature_file}")
    if len(ensemble_analysis) > 0:
        print(f"   📄 {ensemble_file}")
    
    return {
        'best_models_summary': best_models_summary,
        'model_comparison': comparison_df,
        'ensemble_analysis': ensemble_analysis,
        'performance_metrics': performance_df,
        'feature_info': feature_info
    }


def diagnose_corrupted_files(models_dir: str):
    """Diagnose and provide solutions for corrupted joblib files"""
    print(f"\n🔍 DIAGNOSING CORRUPTED FILES in {models_dir}")
    print("=" * 60)
    
    models_path = Path(models_dir)
    
    # Check each joblib file
    joblib_files = ['best_models_info.joblib', 'training_metadata.joblib']
    
    for filename in joblib_files:
        file_path = models_path / filename
        print(f"\n📄 Checking {filename}:")
        
        if not file_path.exists():
            print(f"   ❌ File does not exist")
            continue
        
        # Check file integrity
        is_valid = check_file_integrity(file_path)
        
        if not is_valid:
            print(f"   🔧 Attempting to repair {filename}...")
            
            # Try to create a backup
            backup_path = file_path.with_suffix('.joblib.backup')
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"   💾 Created backup: {backup_path.name}")
            except Exception as e:
                print(f"   ⚠️ Could not create backup: {e}")
            
            # Try to create a minimal replacement
            if filename == 'best_models_info.joblib':
                replacement_data = create_fallback_models_info()
            else:
                replacement_data = create_fallback_metadata()
            
            try:
                joblib.dump(replacement_data, file_path)
                print(f"   ✅ Created minimal replacement for {filename}")
            except Exception as e:
                print(f"   ❌ Could not create replacement: {e}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Re-run the training script to regenerate the files")
    print("2. Check disk space and permissions")
    print("3. Verify the training script completed successfully")
    print("4. Use the --sample-size argument for testing with smaller data")


def main():
    """Main function to run the analysis"""
    
    parser = argparse.ArgumentParser(description="Load and analyze trained models")
    parser.add_argument("--models-dir", type=str, required=True,
                       help="Directory containing best_models_info.joblib and training_metadata.joblib")
    parser.add_argument("--output-dir", type=str, default="./analysis_results",
                       help="Output directory for analysis results (default: ./analysis_results)")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create performance visualizations")
    parser.add_argument("--sample-predictions", action="store_true",
                       help="Make sample predictions (requires dummy data)")
    parser.add_argument("--export-all", action="store_true",
                       help="Export all analysis results to CSV files")
    parser.add_argument("--diagnose", action="store_true",
                       help="Diagnose corrupted files and attempt repair")
    
    args = parser.parse_args()
    
    print("🚀 LOADING AND ANALYZING TRAINED MODELS")
    print("=" * 80)
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run diagnosis if requested
    if args.diagnose:
        diagnose_corrupted_files(args.models_dir)
        return 0
    
    try:
        # Load model files with error handling
        best_models_info, training_metadata = load_model_files_safe(args.models_dir)
        
        if best_models_info is None or training_metadata is None:
            print("❌ Failed to load model files. Exiting.")
            return 1
        
        print(f"✅ Successfully loaded models for {len(best_models_info)} targets")
        
        # Analyze best models info
        best_models_summary = analyze_best_models_info(best_models_info)
        
        # Analyze training metadata
        comparison_df, ensemble_analysis = analyze_training_metadata(training_metadata)
        
        # Load individual models for prediction if requested
        loaded_models = {}
        if args.sample_predictions:
            loaded_models = load_individual_models(best_models_info, args.models_dir)
        
        # Create visualizations if requested
        if args.create_plots:
            plots_dir = create_visualizations(best_models_info, training_metadata, args.output_dir)
        
        # Make sample predictions if requested
        if args.sample_predictions and loaded_models:
            # Create dummy data for demonstration
            feature_cols = training_metadata.get('feature_columns', [])
            if feature_cols:
                print(f"\n🎲 Creating dummy data with {len(feature_cols)} features for prediction demo...")
                
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    col: np.random.normal(0, 1, 100) for col in feature_cols
                })
                
                predictions, pred_summary = predict_with_models(loaded_models, sample_data, args.output_dir)
            else:
                print("⚠️ No feature columns found, skipping sample predictions")
        
        # Export results if requested
        if args.export_all:
            exported_data = export_analysis_results(best_models_info, training_metadata, args.output_dir)
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎉 ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Analyzed {len(best_models_info)} targets")
        print(f"✅ Found {len(training_metadata.get('ensemble_methods_used', []))} ensemble methods")
        print(f"✅ Results saved to: {args.output_dir}")
        
        if args.create_plots:
            print(f"✅ Visualizations saved to: {args.output_dir}/plots/")
        
        print(f"\n📊 QUICK SUMMARY:")
        print("-" * 40)
        for target, info in best_models_info.items():
            target_clean = target.replace('_log1p', '')
            try:
                test_r2 = info['test_metrics']['log']['r2']
                test_wape = info['test_metrics']['orig']['wape']
                print(f"{target_clean:20} → R²: {test_r2:.4f}, WAPE: {test_wape:.3f}")
            except KeyError:
                print(f"{target_clean:20} → Metrics not available")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure the models directory contains both joblib files")
        print("2. Check that the training script completed successfully")
        print("3. Verify file permissions")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
