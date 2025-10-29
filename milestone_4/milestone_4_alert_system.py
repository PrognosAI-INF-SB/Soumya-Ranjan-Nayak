"""Risk Thresholding & Alert System - MILESTONE 4.

This script implements a comprehensive maintenance alert system that:
- Defines multi-level RUL thresholds (Normal, Warning, Critical, Emergency)
- Generates actionable maintenance alerts
- Prioritizes engines by risk level
- Creates visualizations for decision-making
- Exports alerts for maintenance teams
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from enum import Enum

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing required Python packages: pandas and numpy")
    print("Install them with: pip install pandas numpy")
    raise

try:
    import tensorflow as tf
except Exception as e:
    print("Missing required package: tensorflow")
    print("Install it with: pip install tensorflow")
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
OUTPUT_DIR = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_4_alert_system"

# Alert Thresholds (in cycles/hours)
class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = "Normal"
    WARNING = "Warning"
    CRITICAL = "Critical"
    EMERGENCY = "Emergency"

# Configurable thresholds
THRESHOLDS = {
    'emergency': 20,
    'critical': 50,
    'warning': 100,
}


class MaintenanceAlert:
    """Represents a maintenance alert for an engine."""
    
    def __init__(self, engine_id, predicted_rul, actual_rul, alert_level, 
                 recommended_action, time_to_maintenance, confidence):
        self.engine_id = engine_id
        self.predicted_rul = predicted_rul
        self.actual_rul = actual_rul
        self.alert_level = alert_level
        self.recommended_action = recommended_action
        self.time_to_maintenance = time_to_maintenance
        self.confidence = confidence
        self.timestamp = datetime.now()
    
    def to_dict(self):
        """Convert alert to dictionary."""
        return {
            'engine_id': int(self.engine_id),
            'predicted_rul': float(self.predicted_rul),
            'actual_rul': float(self.actual_rul) if self.actual_rul is not None else None,
            'alert_level': self.alert_level.value,
            'recommended_action': self.recommended_action,
            'time_to_maintenance': self.time_to_maintenance,
            'confidence': float(self.confidence),
            'timestamp': self.timestamp.isoformat(),
            'priority_score': self.calculate_priority()
        }
    
    def calculate_priority(self):
        """Calculate priority score (0-100, higher = more urgent)."""
        if self.predicted_rul <= 20:
            base_priority = 95
        elif self.predicted_rul <= 50:
            base_priority = 75
        elif self.predicted_rul <= 100:
            base_priority = 50
        else:
            base_priority = 20
        
        priority = base_priority * self.confidence
        return min(100, max(0, priority))


def load_test_data(test_csv_path, rul_csv_path):
    """Load test data and RUL targets."""
    print(f"Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    
    print(f"Loading RUL targets from: {rul_csv_path}")
    rul_df = pd.read_csv(rul_csv_path)
    
    print(f"[OK] Loaded {len(test_df)} test samples")
    print(f"[OK] Loaded {len(rul_df)} RUL targets")
    
    return test_df, rul_df


def reshape_data_for_lstm(X, sequence_length=30):
    """Reshape flat test data into sequences for LSTM input."""
    n_samples, n_features = X.shape
    unit_ids = X[:, 0].astype(int) if n_features > 24 else None
    
    sequences = []
    engine_ids = []
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
                    engine_ids.append(current_unit)
                unit_data = []
                current_unit = unit
            
            unit_data.append(X[i])
        else:
            if i + sequence_length <= n_samples:
                sequences.append(X[i:i+sequence_length])
                engine_ids.append(i // sequence_length + 1)
    
    if unit_ids is not None and len(unit_data) >= sequence_length:
        features = np.array(unit_data)[:, 2:] if n_features > 24 else np.array(unit_data)
        sequences.append(features[-sequence_length:])
        engine_ids.append(current_unit)
    
    return np.array(sequences), np.array(engine_ids)


def classify_alert_level(rul_prediction):
    """Classify RUL prediction into alert level."""
    if rul_prediction <= THRESHOLDS['emergency']:
        return AlertLevel.EMERGENCY
    elif rul_prediction <= THRESHOLDS['critical']:
        return AlertLevel.CRITICAL
    elif rul_prediction <= THRESHOLDS['warning']:
        return AlertLevel.WARNING
    else:
        return AlertLevel.NORMAL


def get_recommended_action(alert_level, rul):
    """Get recommended maintenance action based on alert level."""
    actions = {
        AlertLevel.EMERGENCY: f"IMMEDIATE ACTION REQUIRED: Ground aircraft and replace engine. Estimated RUL: {rul:.0f} cycles.",
        AlertLevel.CRITICAL: f"Schedule maintenance within next 24-48 hours. Estimated RUL: {rul:.0f} cycles.",
        AlertLevel.WARNING: f"Plan maintenance within next 1-2 weeks. Estimated RUL: {rul:.0f} cycles.",
        AlertLevel.NORMAL: f"Continue normal operations. Monitor regularly. Estimated RUL: {rul:.0f} cycles."
    }
    return actions[alert_level]


def calculate_time_to_maintenance(rul, cycles_per_day=5):
    """Calculate estimated time until maintenance needed."""
    days = rul / cycles_per_day
    
    if days < 1:
        return "Less than 1 day"
    elif days < 7:
        return f"{days:.1f} days"
    elif days < 30:
        weeks = days / 7
        return f"{weeks:.1f} weeks"
    else:
        months = days / 30
        return f"{months:.1f} months"


def estimate_confidence(prediction, historical_rmse=22.43):
    """Estimate prediction confidence based on historical model performance."""
    uncertainty = historical_rmse
    
    if prediction <= 50:
        confidence = max(0.6, 1.0 - (uncertainty / prediction) if prediction > 0 else 0.6)
    else:
        confidence = max(0.7, 1.0 - (uncertainty / 100))
    
    return min(0.95, confidence)


def generate_alerts(predictions, engine_ids, actual_rul=None):
    """Generate maintenance alerts for all engines."""
    alerts = []
    
    for idx, (engine_id, pred_rul) in enumerate(zip(engine_ids, predictions)):
        alert_level = classify_alert_level(pred_rul)
        recommended_action = get_recommended_action(alert_level, pred_rul)
        time_to_maintenance = calculate_time_to_maintenance(pred_rul)
        confidence = estimate_confidence(pred_rul)
        
        actual = actual_rul[idx] if actual_rul is not None and idx < len(actual_rul) else None
        
        alert = MaintenanceAlert(
            engine_id=engine_id,
            predicted_rul=pred_rul,
            actual_rul=actual,
            alert_level=alert_level,
            recommended_action=recommended_action,
            time_to_maintenance=time_to_maintenance,
            confidence=confidence
        )
        
        alerts.append(alert)
    
    return alerts


def create_alert_visualizations(alerts, output_dir):
    """Create comprehensive visualizations for the alert system."""
    os.makedirs(output_dir, exist_ok=True)
    
    alert_data = [a.to_dict() for a in alerts]
    df = pd.DataFrame(alert_data)
    
    sns.set_style("whitegrid")
    colors = {'Emergency': '#d32f2f', 'Critical': '#f57c00', 'Warning': '#fbc02d', 'Normal': '#388e3c'}
    
    # 1. Alert Level Distribution
    plt.figure(figsize=(10, 6))
    alert_counts = df['alert_level'].value_counts()
    bars = plt.bar(alert_counts.index, alert_counts.values, 
                   color=[colors.get(level, '#999') for level in alert_counts.index])
    plt.xlabel('Alert Level', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Engines', fontsize=12, fontweight='bold')
    plt.title('Alert Level Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alert_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RUL Distribution by Alert Level
    plt.figure(figsize=(12, 6))
    alert_order = ['Emergency', 'Critical', 'Warning', 'Normal']
    alert_colors = [colors.get(level, '#999') for level in alert_order]
    
    df_sorted = df.copy()
    df_sorted['alert_level'] = pd.Categorical(df_sorted['alert_level'], 
                                               categories=alert_order, ordered=True)
    
    sns.boxplot(data=df_sorted, x='alert_level', y='predicted_rul', 
                palette=alert_colors, order=alert_order)
    plt.xlabel('Alert Level', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted RUL (cycles)', fontsize=12, fontweight='bold')
    plt.title('RUL Distribution by Alert Level', fontsize=14, fontweight='bold')
    
    plt.axhline(y=THRESHOLDS['emergency'], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f"Emergency ({THRESHOLDS['emergency']})")
    plt.axhline(y=THRESHOLDS['critical'], color='orange', linestyle='--', 
                linewidth=2, alpha=0.7, label=f"Critical ({THRESHOLDS['critical']})")
    plt.axhline(y=THRESHOLDS['warning'], color='yellow', linestyle='--', 
                linewidth=2, alpha=0.7, label=f"Warning ({THRESHOLDS['warning']})")
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rul_by_alert_level.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top 10 High-Risk Engines
    top_risk = df.nlargest(10, 'priority_score')
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(top_risk)), top_risk['priority_score'], 
                    color=[colors.get(level, '#999') for level in top_risk['alert_level']])
    plt.yticks(range(len(top_risk)), 
              [f"Engine {int(eid)}" for eid in top_risk['engine_id']])
    plt.xlabel('Priority Score', fontsize=12, fontweight='bold')
    plt.ylabel('Engine ID', fontsize=12, fontweight='bold')
    plt.title('Top 10 High-Risk Engines', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for idx, (i, row) in enumerate(top_risk.iterrows()):
        plt.text(row['priority_score'] + 2, idx, 
                f"RUL: {row['predicted_rul']:.0f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_risk_engines.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] All visualizations saved to: {output_dir}")


def generate_alert_report(alerts, output_dir, threshold_config):
    """Generate comprehensive alert report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'alert_report.txt')
    
    emergency_alerts = [a for a in alerts if a.alert_level == AlertLevel.EMERGENCY]
    critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
    warning_alerts = [a for a in alerts if a.alert_level == AlertLevel.WARNING]
    normal_engines = [a for a in alerts if a.alert_level == AlertLevel.NORMAL]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("MAINTENANCE ALERT SYSTEM REPORT - MILESTONE 4\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Engines Monitored: {len(alerts)}\n\n")
        
        f.write("-" * 90 + "\n")
        f.write("ALERT THRESHOLD CONFIGURATION\n")
        f.write("-" * 90 + "\n")
        f.write(f"Emergency Alert:  RUL <= {threshold_config['emergency']} cycles\n")
        f.write(f"Critical Alert:   RUL <= {threshold_config['critical']} cycles\n")
        f.write(f"Warning Alert:    RUL <= {threshold_config['warning']} cycles\n")
        f.write(f"Normal Operation: RUL >  {threshold_config['warning']} cycles\n\n")
        
        f.write("-" * 90 + "\n")
        f.write("ALERT SUMMARY\n")
        f.write("-" * 90 + "\n")
        f.write(f"Emergency Alerts:  {len(emergency_alerts):>3} ({100*len(emergency_alerts)/len(alerts):>5.1f}%)\n")
        f.write(f"Critical Alerts:   {len(critical_alerts):>3} ({100*len(critical_alerts)/len(alerts):>5.1f}%)\n")
        f.write(f"Warning Alerts:    {len(warning_alerts):>3} ({100*len(warning_alerts)/len(alerts):>5.1f}%)\n")
        f.write(f"Normal Operations: {len(normal_engines):>3} ({100*len(normal_engines)/len(alerts):>5.1f}%)\n\n")
        
        if emergency_alerts:
            f.write("=" * 90 + "\n")
            f.write("EMERGENCY ALERTS - IMMEDIATE ACTION REQUIRED\n")
            f.write("=" * 90 + "\n\n")
            
            emergency_alerts.sort(key=lambda x: x.calculate_priority(), reverse=True)
            
            for alert in emergency_alerts[:5]:
                f.write(f"Engine ID: {alert.engine_id}\n")
                f.write(f"  Predicted RUL:     {alert.predicted_rul:.2f} cycles\n")
                f.write(f"  Priority Score:    {alert.calculate_priority():.1f}/100\n")
                f.write(f"  Confidence:        {alert.confidence*100:.1f}%\n")
                f.write(f"  Action Required:   {alert.recommended_action}\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("END OF ALERT REPORT\n")
        f.write("=" * 90 + "\n")
    
    print(f"[OK] Alert report saved to: {report_path}")


def export_alerts_csv(alerts, output_dir):
    """Export alerts to CSV for maintenance teams."""
    os.makedirs(output_dir, exist_ok=True)
    
    alert_data = [a.to_dict() for a in alerts]
    df = pd.DataFrame(alert_data)
    df = df.sort_values('priority_score', ascending=False)
    
    csv_path = os.path.join(output_dir, 'maintenance_alerts_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"[OK] All alerts exported to: {csv_path}")
    
    high_priority = df[df['alert_level'].isin(['Emergency', 'Critical'])]
    if len(high_priority) > 0:
        priority_path = os.path.join(output_dir, 'maintenance_alerts_priority.csv')
        high_priority.to_csv(priority_path, index=False)
        print(f"[OK] Priority alerts exported to: {priority_path}")


def export_alerts_json(alerts, output_dir, threshold_config):
    """Export alerts to JSON for API integration."""
    os.makedirs(output_dir, exist_ok=True)
    
    alert_data = [a.to_dict() for a in alerts]
    alert_data.sort(key=lambda x: x['priority_score'], reverse=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_engines': len(alerts),
        'threshold_config': threshold_config,
        'summary': {
            'emergency': sum(1 for a in alert_data if a['alert_level'] == 'Emergency'),
            'critical': sum(1 for a in alert_data if a['alert_level'] == 'Critical'),
            'warning': sum(1 for a in alert_data if a['alert_level'] == 'Warning'),
            'normal': sum(1 for a in alert_data if a['alert_level'] == 'Normal')
        },
        'alerts': alert_data
    }
    
    json_path = os.path.join(output_dir, 'alerts_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    
    print(f"[OK] Alerts JSON saved to: {json_path}")


def print_alert_summary(alerts):
    """Print alert summary to console."""
    print("\n" + "=" * 90)
    print("ALERT SUMMARY")
    print("=" * 90)
    
    emergency = sum(1 for a in alerts if a.alert_level == AlertLevel.EMERGENCY)
    critical = sum(1 for a in alerts if a.alert_level == AlertLevel.CRITICAL)
    warning = sum(1 for a in alerts if a.alert_level == AlertLevel.WARNING)
    normal = sum(1 for a in alerts if a.alert_level == AlertLevel.NORMAL)
    
    print(f"Emergency Alerts:  {emergency:>3} ({100*emergency/len(alerts):>5.1f}%)")
    print(f"Critical Alerts:   {critical:>3} ({100*critical/len(alerts):>5.1f}%)")
    print(f"Warning Alerts:    {warning:>3} ({100*warning/len(alerts):>5.1f}%)")
    print(f"Normal Operations: {normal:>3} ({100*normal/len(alerts):>5.1f}%)")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Risk Thresholding & Alert System - Milestone 4")
    parser.add_argument('--test-csv', default=TEST_CSV, help='Path to test features CSV')
    parser.add_argument('--rul-csv', default=RUL_CSV, help='Path to RUL targets CSV')
    parser.add_argument('--model-path', default=MODEL_PATH, help='Path to saved model')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for LSTM')
    parser.add_argument('--emergency-threshold', type=int, default=20, help='Emergency alert threshold')
    parser.add_argument('--critical-threshold', type=int, default=50, help='Critical alert threshold')
    parser.add_argument('--warning-threshold', type=int, default=100, help='Warning alert threshold')

    args = parser.parse_args()
    
    THRESHOLDS['emergency'] = args.emergency_threshold
    THRESHOLDS['critical'] = args.critical_threshold
    THRESHOLDS['warning'] = args.warning_threshold

    print("\n" + "=" * 90)
    print("MILESTONE 4: RISK THRESHOLDING & ALERT SYSTEM")
    print("=" * 90 + "\n")
    
    # Load data
    test_df, rul_df = load_test_data(args.test_csv, args.rul_csv)
    
    # Prepare features
    X_test = test_df.values
    print(f"Reshaping data into sequences (length={args.sequence_length})...")
    X_sequences, engine_ids = reshape_data_for_lstm(X_test, args.sequence_length)
    print(f"[OK] Created {len(X_sequences)} sequences for {len(np.unique(engine_ids))} engines")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    print("[OK] Model loaded successfully")
    
    # Make predictions
    print("\nGenerating RUL predictions...")
    predictions = model.predict(X_sequences, verbose=0).squeeze()
    print(f"[OK] Generated predictions for {len(predictions)} engines")
    
    # Get actual RUL values
    actual_rul = rul_df['RUL'].values if 'RUL' in rul_df.columns else None
    
    # Generate alerts
    print("\nGenerating maintenance alerts...")
    alerts = generate_alerts(predictions, engine_ids, actual_rul)
    print(f"[OK] Generated {len(alerts)} alerts")
    
    # Print summary
    print_alert_summary(alerts)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_alert_visualizations(alerts, args.output_dir)
    
    # Generate reports
    print("\nGenerating alert report...")
    generate_alert_report(alerts, args.output_dir, THRESHOLDS)
    
    # Export data
    print("\nExporting alert data...")
    export_alerts_csv(alerts, args.output_dir)
    export_alerts_json(alerts, args.output_dir, THRESHOLDS)
    
    print("\n" + "=" * 90)
    print("MILESTONE 4 COMPLETE!")
    print("=" * 90)
    print(f"\nAll deliverables saved to: {args.output_dir}")
    print("  • alert_report.txt - Detailed alert report")
    print("  • maintenance_alerts_all.csv - All alerts")
    print("  • maintenance_alerts_priority.csv - High-priority alerts")
    print("  • alerts_data.json - JSON format for API integration")
    print("  • alert_distribution.png - Alert level distribution")
    print("  • rul_by_alert_level.png - RUL by alert level")
    print("  • top_risk_engines.png - Top 10 high-risk engines")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()