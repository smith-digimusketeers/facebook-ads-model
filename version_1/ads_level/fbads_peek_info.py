#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load and Use Trained Models Script
==================================
Analyzes and uses models from ensemble training output.
Loads best_models_info.joblib and training_metadata.joblib files.
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

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_model_files(models_dir: str):
    """Load both joblib files from the models directory"""
    
    models_path = Path(models_dir)
    
    # Check if directory exists
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load best_models_info.joblib
    best_models_path = models_path / 'best_models_info.joblib'
    if not best_models_path.exists():
        raise FileNotFoundError(f"best_models_info.joblib not found in {models_dir}")
    
    print(f"📂 Loading best_models_info.joblib from {models_dir}...")
    best_models_info = joblib.load(best_models_path)
    
    # Load training_metadata.joblib
    metadata_path = models_path / 'training_metadata.joblib'
    if not metadata_path.exists():
        raise FileNotFoundError(f"training_metadata.joblib not found in {models_dir}")
    
    print(f"📂 Loading training_metadata.joblib from {models_dir}...")
    training_metadata = joblib.load(metadata_path)
    
    return best_models_info, training_metadata


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
        val_r2 = info['val_metrics']['log']['r2']
        test_r2 = info['test_metrics']['log']['r2']
        model_name = info['model_name']
        
        summary_data.append({
            'Target': target_clean,
            'Best Model': model_name,
            'Val R²': val_r2,
            'Test R²': test_r2,
            'Model Path': info['model_path']
        })
        
        print(f"  {target_clean:20} → {model_name:20} (Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f})")
    
    # Detailed metrics for each target
    print(f"\n📈 DETAILED METRICS BY TARGET:")
    print("=" * 80)
    
    for target, info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        print(f"\n🎯 {target_clean.upper()}")
        print("-" * 40)
        
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
        
        # Model info
        print(f"Model: {info['model_name']}")
        print(f"Path: {info['model_path']}")
    
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
    print(f"  Feature columns: {len(training_metadata['feature_columns'])}")
    print(f"  Target columns: {len(training_metadata['target_columns'])}")
    print(f"  Ensemble methods: {training_metadata['ensemble_methods_used']}")
    print(f"  Batch size: {training_metadata.get('batch_size', 'Auto')}")
    print(f"  Sample size: {training_metadata.get('sample_size', 'Full dataset')}")
    print(f"  Quick mode: {training_metadata.get('quick_mode', False)}")
    
    # Model comparison analysis
    print(f"\n📊 MODEL COMPARISON ANALYSIS:")
    print("=" * 80)
    
    comparison_data = []
    
    for target, results in training_metadata['all_results'].items():
        target_clean = target.replace('_log1p', '')
        
        # Individual models
        if 'model_results' in results:
            for model_name, metrics in results['model_results'].items():
                comparison_data.append({
                    'Target': target_clean,
                    'Model': model_name,
                    'Type': 'Individual',
                    'Val R²': metrics['val_r2'],
                    'Val MAE': metrics['val_mae'],
                    'Train Time (s)': metrics.get('train_time', 0)
                })
        
        # Ensemble models
        if 'ensemble_results' in results and results['ensemble_results']:
            for model_name, metrics in results['ensemble_results'].items():
                comparison_data.append({
                    'Target': target_clean,
                    'Model': f"ensemble_{model_name}",
                    'Type': 'Ensemble',
                    'Val R²': metrics['val_r2'],
                    'Val MAE': metrics['val_mae'],
                    'Train Time (s)': 0
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Show best model per target
    print("\n🏆 BEST MODEL PER TARGET:")
    print("-" * 60)
    for target in comparison_df['Target'].unique():
        target_df = comparison_df[comparison_df['Target'] == target]
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
    
    # 1. Performance comparison across targets
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Across Targets', fontsize=16)
    
    targets = list(best_models_info.keys())
    target_names = [t.replace('_log1p', '') for t in targets]
    
    # R² scores
    val_r2_scores = [best_models_info[t]['val_metrics']['log']['r2'] for t in targets]
    test_r2_scores = [best_models_info[t]['test_metrics']['log']['r2'] for t in targets]
    
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
    val_wape_scores = [best_models_info[t]['val_metrics']['orig']['wape'] for t in targets]
    test_wape_scores = [best_models_info[t]['test_metrics']['orig']['wape'] for t in targets]
    
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
    model_types = [best_models_info[t]['model_name'] for t in targets]
    model_type_counts = pd.Series(model_types).value_counts()
    
    axes[1, 0].pie(model_type_counts.values, labels=model_type_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Distribution of Best Model Types')
    
    # Performance heatmap
    metrics = ['val_r2', 'test_r2', 'val_wape', 'test_wape']
    metric_names = ['Val R²', 'Test R²', 'Val WAPE', 'Test WAPE']
    
    heatmap_data = []
    for target in targets:
        row = [
            best_models_info[target]['val_metrics']['log']['r2'],
            best_models_info[target]['test_metrics']['log']['r2'],
            best_models_info[target]['val_metrics']['orig']['wape'],
            best_models_info[target]['test_metrics']['orig']['wape']
        ]
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
    
    # 2. Ensemble vs Individual comparison
    comparison_df, ensemble_analysis = analyze_training_metadata(training_metadata)
    
    if len(ensemble_analysis) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Ensemble vs Individual Model Performance', fontsize=16)
        
        # Improvement percentages
        improvements = ensemble_analysis['Improvement (%)']
        targets = ensemble_analysis['Target']
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax1.bar(targets, improvements, color=colors, alpha=0.7)
        ax1.set_xlabel('Targets')
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('Ensemble Performance Improvement')
        ax1.set_xticklabels(targets, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                    f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # R² comparison
        individual_r2 = ensemble_analysis['Best Individual']
        ensemble_r2 = ensemble_analysis['Best Ensemble']
        
        x = np.arange(len(targets))
        width = 0.35
        
        ax2.bar(x - width/2, individual_r2, width, label='Best Individual', alpha=0.8)
        ax2.bar(x + width/2, ensemble_r2, width, label='Best Ensemble', alpha=0.8)
        ax2.set_xlabel('Targets')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(targets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved visualizations to {plots_dir}/")
    return plots_dir


def predict_with_models(best_models_info: Dict, sample_data: pd.DataFrame, output_dir: str):
    """Make predictions using all trained models"""
    
    print(f"\n🔮 MAKING PREDICTIONS WITH ALL MODELS")
    print("=" * 60)
    
    predictions = {}
    prediction_summary = []
    
    for target, model_info in best_models_info.items():
        target_clean = target.replace('_log1p', '')
        print(f"🎯 Predicting {target_clean}...")
        
        try:
            # Load model
            model = joblib.load(model_info['model_path'])
            
            # Make predictions
            pred_log = model.predict(sample_data)
            pred_orig = np.expm1(pred_log)  # Convert to original scale
            
            predictions[target_clean] = {
                'predictions_log': pred_log,
                'predictions_original': pred_orig,
                'model_name': model_info['model_name']
            }
            
            # Summary stats
            summary = {
                'Target': target_clean,
                'Model': model_info['model_name'],
                'Mean Prediction': pred_orig.mean(),
                'Std Prediction': pred_orig.std(),
                'Min Prediction': pred_orig.min(),
                'Max Prediction': pred_orig.max(),
                'Sample Size': len(pred_orig)
            }
            prediction_summary.append(summary)
            
            print(f"   ✅ {target_clean}: Mean = {pred_orig.mean():,.0f}, Range = {pred_orig.min():,.0f} to {pred_orig.max():,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error with {target_clean}: {e}")
    
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
    
    performance_df = pd.DataFrame(performance_data)
    performance_file = Path(output_dir) / "performance_metrics.csv"
    performance_df.to_csv(performance_file, index=False)
    
    # 5. Feature information
    feature_info = {
        'feature_columns': training_metadata['feature_columns'],
        'target_columns': training_metadata['target_columns'],
        'total_features': len(training_metadata['feature_columns']),
        'total_targets': len(training_metadata['target_columns']),
        'ensemble_methods': training_metadata['ensemble_methods_used'],
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
        print(f"   �� {ensemble_file}")
    
    return {
        'best_models_summary': best_models_summary,
        'model_comparison': comparison_df,
        'ensemble_analysis': ensemble_analysis,
        'performance_metrics': performance_df,
        'feature_info': feature_info
    }


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
    
    args = parser.parse_args()
    
    print("🚀 LOADING AND ANALYZING TRAINED MODELS")
    print("=" * 80)
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load model files
        best_models_info, training_metadata = load_model_files(args.models_dir)
        
        print(f"✅ Successfully loaded models for {len(best_models_info)} targets")
        
        # Analyze best models info
        best_models_summary = analyze_best_models_info(best_models_info)
        
        # Analyze training metadata
        comparison_df, ensemble_analysis = analyze_training_metadata(training_metadata)
        
        # Create visualizations if requested
        if args.create_plots:
            plots_dir = create_visualizations(best_models_info, training_metadata, args.output_dir)
        
        # Make sample predictions if requested
        if args.sample_predictions:
            # Create dummy data for demonstration
            feature_cols = training_metadata['feature_columns']
            print(f"\n🎲 Creating dummy data with {len(feature_cols)} features for prediction demo...")
            
            np.random.seed(42)
            sample_data = pd.DataFrame({
                col: np.random.normal(0, 1, 100) for col in feature_cols
            })
            
            predictions, pred_summary = predict_with_models(best_models_info, sample_data, args.output_dir)
        
        # Export results if requested
        if args.export_all:
            exported_data = export_analysis_results(best_models_info, training_metadata, args.output_dir)
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎉 ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"✅ Analyzed {len(best_models_info)} targets")
        print(f"✅ Found {len(training_metadata['ensemble_methods_used'])} ensemble methods")
        print(f"✅ Results saved to: {args.output_dir}")
        
        if args.create_plots:
            print(f"✅ Visualizations saved to: {args.output_dir}/plots/")
        
        print(f"\n📊 QUICK SUMMARY:")
        print("-" * 40)
        for target, info in best_models_info.items():
            target_clean = target.replace('_log1p', '')
            test_r2 = info['test_metrics']['log']['r2']
            test_wape = info['test_metrics']['orig']['wape']
            print(f"{target_clean:20} → R²: {test_r2:.4f}, WAPE: {test_wape:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())