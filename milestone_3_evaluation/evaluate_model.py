"""Evaluate a trained model on test data - MILESTONE 3 COMPLETE.

This script provides comprehensive model evaluation including:
- Multiple performance metrics (RMSE, MSE, MAE, R²)
- Visualization plots (predicted vs actual, residuals, error distribution)
- Bias and error analysis
- Detailed evaluation report generation
"""

import sys
import os
import argparse
import json
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing required Python packages: pandas and numpy")
    print("Install them with: pip install pandas numpy")
    raise

try:
    import tensorflow as tf
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except Exception as e:
    print("Missing required packages: tensorflow and scikit-learn")
    print("Install them with: pip install tensorflow scikit-learn")
    raise

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Missing required packages for visualization: matplotlib and seaborn")
    print("Install them with: pip install matplotlib seaborn")
    raise

# Default paths
TEST_CSV = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_1_data_preparation\processed_data\test_processed_FD001.csv"
RUL_CSV = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_1_data_preparation\processed_data\rul_targets_FD001.csv"
MODEL_PATH = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_2_model_training\saved_models\lstm_rul_model_FD001.h5"
OUTPUT_DIR = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_3_evaluation"

# Evaluation thresholds
RMSE_THRESHOLD = 30.0  # Adjust based on your requirements


def load_model_any(path: str):
    """Try to load a model either via tf.keras.models.load_model or tf.saved_model.load."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path does not exist: {path}")

    try:
        model = tf.keras.models.load_model(path, compile=False)
        print("Model loaded successfully (without compilation)")
        return model
    except Exception as e:
        print(f"Failed to load with keras loader: {e}")
        try:
            loaded = tf.saved_model.load(path)
            if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
                print("Loaded as SavedModel with serving_default signature")
                return loaded.signatures['serving_default']
            print("Loaded as SavedModel")
            return loaded
        except Exception as e2:
            print(f"Failed to load as SavedModel: {e2}")
            raise


def predict_with_model(model, X: np.ndarray) -> np.ndarray:
    """Make predictions for a numpy array X using the provided model object."""
    if hasattr(model, 'predict'):
        preds = model.predict(X)
    else:
        tf_X = tf.convert_to_tensor(X, dtype=tf.float32)
        out = model(tf_X)
        if isinstance(out, dict):
            out = list(out.values())[0]
        preds = out.numpy()
    return np.asarray(preds).squeeze()


def reshape_data_for_lstm(X, sequence_length=30):
    """Reshape flat test data into sequences for LSTM input."""
    n_samples, n_features = X.shape
    unit_ids = X[:, 0].astype(int) if n_features > 24 else None
    
    sequences = []
    current_unit = None
    unit_data = []
    
    for i in range(n_samples):
        if unit_ids is not None:
            unit = unit_ids[i]
            if current_unit is None:
                current_unit = unit
            
            if unit != current_unit:
                if len(unit_data) >= sequence_length:
                    features = np.array(unit_data)[:, 2:] if n_features > 24 else np.array(unit_data)
                    sequences.append(features[-sequence_length:])
                unit_data = []
                current_unit = unit
            
            unit_data.append(X[i])
        else:
            if i + sequence_length <= n_samples:
                sequences.append(X[i:i+sequence_length])
    
    if unit_ids is not None and len(unit_data) >= sequence_length:
        features = np.array(unit_data)[:, 2:] if n_features > 24 else np.array(unit_data)
        sequences.append(features[-sequence_length:])
    
    return np.array(sequences)


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # +epsilon to avoid div by zero
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
        'max_error': max_error
    }


def analyze_errors_by_rul_range(y_true, y_pred):
    """Analyze model performance across different RUL ranges."""
    ranges = [(0, 50), (50, 100), (100, 150), (150, float('inf'))]
    range_analysis = []
    
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            range_true = y_true[mask]
            range_pred = y_pred[mask]
            
            range_metrics = {
                'range': f'{low}-{high if high != float("inf") else "inf"}',
                'count': np.sum(mask),
                'rmse': np.sqrt(mean_squared_error(range_true, range_pred)),
                'mae': mean_absolute_error(range_true, range_pred),
                'mean_error': np.mean(range_pred - range_true),
                'std_error': np.std(range_pred - range_true)
            }
            range_analysis.append(range_metrics)
    
    return range_analysis


def create_visualizations(y_true, y_pred, output_dir):
    """Create comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Predicted vs Actual Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual RUL', fontsize=12)
    plt.ylabel('Predicted RUL', fontsize=12)
    plt.title('Predicted vs Actual RUL', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residual Plot
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Actual RUL', fontsize=12)
    plt.ylabel('Residual (Predicted - Actual)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Residual (Predicted - Actual)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(residuals, vert=True)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_ylabel('Residual (Predicted - Actual)', fontsize=11)
    axes[1].set_title('Error Box Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Time Series Plot (first 5 engines)
    n_engines = min(5, len(y_true))
    samples_per_engine = len(y_true) // n_engines if n_engines > 0 else len(y_true)
    
    plt.figure(figsize=(14, 6))
    for i in range(n_engines):
        start_idx = i * samples_per_engine
        end_idx = (i + 1) * samples_per_engine
        x_range = range(start_idx, end_idx)
        plt.plot(x_range, y_true[start_idx:end_idx], 'o-', alpha=0.6, label=f'Engine {i+1} Actual')
        plt.plot(x_range, y_pred[start_idx:end_idx], 's--', alpha=0.6, label=f'Engine {i+1} Predicted')
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('RUL', fontsize=12)
    plt.title('RUL Predictions Over Time (Sample Engines)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Error by RUL Range
    ranges = [(0, 50), (50, 100), (100, 150), (150, 250)]
    range_errors = []
    range_labels = []
    
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            range_errors.append(np.abs(residuals[mask]))
            range_labels.append(f'{low}-{high}')
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(range_errors, tick_labels=range_labels)
    plt.xlabel('RUL Range', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Error Distribution by RUL Range', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_rul_range.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] All visualization plots saved to: {output_dir}")


def generate_evaluation_report(metrics, range_analysis, y_true, y_pred, 
                               model_path, test_csv, output_dir, threshold):
    """Generate a comprehensive evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT - MILESTONE 3\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Data: {test_csv}\n")
        f.write(f"Total Samples: {len(y_true)}\n\n")
        
        # Overall Metrics
        f.write("-" * 80 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"Mean Squared Error (MSE):       {metrics['mse']:.4f}\n")
        f.write(f"Mean Absolute Error (MAE):      {metrics['mae']:.4f}\n")
        f.write(f"R² Score:                       {metrics['r2_score']:.4f}\n")
        f.write(f"Mean Absolute % Error (MAPE):   {metrics['mape']:.2f}%\n")
        f.write(f"Maximum Error:                  {metrics['max_error']:.4f}\n\n")
        
        # Threshold evaluation
        f.write("-" * 80 + "\n")
        f.write("THRESHOLD EVALUATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Predefined RMSE Threshold: {threshold:.2f}\n")
        f.write(f"Achieved RMSE:             {metrics['rmse']:.4f}\n")
        if metrics['rmse'] <= threshold:
            f.write(f"Status: [PASSED] RMSE below threshold\n\n")
        else:
            f.write(f"Status: [FAILED] RMSE exceeds threshold by {metrics['rmse'] - threshold:.4f}\n\n")
        
        # Error Analysis
        residuals = y_pred - y_true
        f.write("-" * 80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Error (Bias):        {np.mean(residuals):.4f}\n")
        f.write(f"Std Dev of Errors:        {np.std(residuals):.4f}\n")
        f.write(f"Median Absolute Error:    {np.median(np.abs(residuals)):.4f}\n")
        f.write(f"95th Percentile Error:    {np.percentile(np.abs(residuals), 95):.4f}\n\n")
        
        # Bias identification
        if np.mean(residuals) > 5:
            f.write("WARNING: MODEL BIAS - Tendency to OVER-predict RUL\n\n")
        elif np.mean(residuals) < -5:
            f.write("WARNING: MODEL BIAS - Tendency to UNDER-predict RUL\n\n")
        else:
            f.write("INFO: No significant systematic bias detected\n\n")
        
        # Performance by RUL Range
        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE BY RUL RANGE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Range':<15} {'Count':<10} {'RMSE':<12} {'MAE':<12} {'Mean Error':<15} {'Std Error':<12}\n")
        f.write("-" * 80 + "\n")
        
        for ra in range_analysis:
            f.write(f"{ra['range']:<15} {ra['count']:<10} {ra['rmse']:<12.4f} "
                   f"{ra['mae']:<12.4f} {ra['mean_error']:<15.4f} {ra['std_error']:<12.4f}\n")
        
        f.write("\n")
        
        # Model Limitations
        f.write("-" * 80 + "\n")
        f.write("MODEL LIMITATIONS & INSIGHTS\n")
        f.write("-" * 80 + "\n")
        
        # Identify challenging ranges
        worst_range = max(range_analysis, key=lambda x: x['rmse'])
        f.write(f"• Highest error in RUL range: {worst_range['range']} (RMSE: {worst_range['rmse']:.4f})\n")
        
        # Check for outliers
        outlier_threshold = 3 * np.std(residuals)
        outliers = np.sum(np.abs(residuals) > outlier_threshold)
        f.write(f"• Number of outlier predictions (>3σ): {outliers} ({100*outliers/len(residuals):.2f}%)\n")
        
        # Early vs late life performance
        early_mask = y_true > 100
        late_mask = y_true <= 50
        if np.sum(early_mask) > 0 and np.sum(late_mask) > 0:
            early_rmse = np.sqrt(mean_squared_error(y_true[early_mask], y_pred[early_mask]))
            late_rmse = np.sqrt(mean_squared_error(y_true[late_mask], y_pred[late_mask]))
            f.write(f"• Early life (RUL>100) RMSE: {early_rmse:.4f}\n")
            f.write(f"• Late life (RUL≤50) RMSE:   {late_rmse:.4f}\n")
            
            if early_rmse > late_rmse * 1.5:
                f.write("  → Model performs worse in early life predictions\n")
            elif late_rmse > early_rmse * 1.5:
                f.write("  → Model performs worse in late life predictions\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        if metrics['rmse'] > threshold:
            f.write("• Consider additional hyperparameter tuning\n")
            f.write("• Evaluate feature engineering improvements\n")
            f.write("• Try ensemble methods or more complex architectures\n")
        
        if np.abs(np.mean(residuals)) > 5:
            f.write("• Address systematic bias through model calibration\n")
        
        if outliers > len(residuals) * 0.05:
            f.write("• Investigate outlier predictions for data quality issues\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[OK] Evaluation report saved to: {report_path}")
    
    # Also save metrics as JSON
    json_path = os.path.join(output_dir, 'metrics.json')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'test_csv': test_csv,
        'overall_metrics': convert_numpy_types(metrics),
        'threshold': float(threshold),
        'passed_threshold': bool(metrics['rmse'] <= threshold),
        'range_analysis': convert_numpy_types(range_analysis),
        'sample_count': int(len(y_true))
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"[OK] Metrics JSON saved to: {json_path}")


def save_predictions(y_true, y_pred, output_dir):
    """Save predictions to CSV for further analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'actual_rul': y_true,
        'predicted_rul': y_pred,
        'error': y_pred - y_true,
        'absolute_error': np.abs(y_pred - y_true)
    })
    
    csv_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Predictions saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation - Milestone 3")
    parser.add_argument('--test-csv', default=TEST_CSV, help='Path to test features CSV')
    parser.add_argument('--rul-csv', default=RUL_CSV, help='Path to RUL targets CSV')
    parser.add_argument('--model-path', default=MODEL_PATH, help='Path to saved model')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for LSTM')
    parser.add_argument('--rmse-threshold', type=float, default=RMSE_THRESHOLD, 
                       help='RMSE threshold for evaluation')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MILESTONE 3: MODEL EVALUATION & PERFORMANCE ASSESSMENT")
    print("=" * 80 + "\n")
    
    print(f"Loading test data from: {args.test_csv}")
    print(f"Loading RUL targets from: {args.rul_csv}")
    print(f"Loading model from: {args.model_path}")
    print(f"Output directory: {args.output_dir}\n")

    # Load data
    X_test = pd.read_csv(args.test_csv)
    y_test = pd.read_csv(args.rul_csv).values.flatten()

    print(f"Test data shape (flat): {X_test.shape}")
    print(f"Target data shape: {y_test.shape}\n")

    # Reshape data
    X_flat = X_test.values
    print(f"Reshaping data into sequences of length {args.sequence_length}...")
    X_np = reshape_data_for_lstm(X_flat, sequence_length=args.sequence_length)
    print(f"Reshaped data shape: {X_np.shape}\n")

    # Load model
    model = load_model_any(args.model_path)
    print()

    # Predict
    print("Making predictions...")
    predictions = predict_with_model(model, X_np)
    print(f"Predictions shape: {predictions.shape}\n")

    # Handle shape mismatch
    if predictions.shape[0] != y_test.shape[0]:
        print(f"WARNING: Shape mismatch detected - aligning to minimum length")
        min_len = min(predictions.shape[0], y_test.shape[0])
        predictions = predictions[:min_len]
        y_test = y_test[:min_len]
        print(f"Using {min_len} samples\n")

    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(y_test, predictions)
    
    # Analyze errors by RUL range
    print("Analyzing errors by RUL range...")
    range_analysis = analyze_errors_by_rul_range(y_test, predictions)
    
    # Create visualizations
    print("Generating visualization plots...")
    create_visualizations(y_test, predictions, args.output_dir)
    
    # Save predictions
    print("Saving predictions to CSV...")
    save_predictions(y_test, predictions, args.output_dir)
    
    # Generate report
    print("Generating evaluation report...")
    generate_evaluation_report(metrics, range_analysis, y_test, predictions,
                              args.model_path, args.test_csv, args.output_dir,
                              args.rmse_threshold)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"RMSE:     {metrics['rmse']:.4f}")
    print(f"MAE:      {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Threshold: {args.rmse_threshold:.2f}")
    
    if metrics['rmse'] <= args.rmse_threshold:
        print(f"\n[SUCCESS] MODEL PASSED: RMSE is below threshold")
    else:
        print(f"\n[WARNING] MODEL FAILED: RMSE exceeds threshold by {metrics['rmse'] - args.rmse_threshold:.4f}")
    
    print("\n" + "=" * 80)
    print("MILESTONE 3 COMPLETE!")
    print("=" * 80)
    print(f"\nAll deliverables saved to: {args.output_dir}")
    print("  • evaluation_report.txt - Detailed analysis")
    print("  • metrics.json - Machine-readable results")
    print("  • predictions.csv - All predictions with errors")
    print("  • predicted_vs_actual.png - Scatter plot")
    print("  • residual_plot.png - Residual analysis")
    print("  • error_distribution.png - Error histogram and box plot")
    print("  • time_series_comparison.png - Sample predictions over time")
    print("  • error_by_rul_range.png - Performance by RUL range")
    print("\n")


if __name__ == '__main__':
    main()