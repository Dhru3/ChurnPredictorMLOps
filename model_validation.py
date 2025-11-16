#!/usr/bin/env python3
"""
Automated Model Validation System
Validates model performance, fairness, and robustness before deployment
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Comprehensive model validation system
    Checks: Performance, Fairness, Robustness
    """
    
    def __init__(self, 
                 min_accuracy: float = 0.75,
                 min_precision: float = 0.70,
                 min_recall: float = 0.65,
                 min_f1: float = 0.70,
                 min_roc_auc: float = 0.80,
                 fairness_threshold: float = 0.10):
        """
        Initialize validator with thresholds
        
        Args:
            min_accuracy: Minimum acceptable accuracy
            min_precision: Minimum acceptable precision
            min_recall: Minimum acceptable recall
            min_f1: Minimum acceptable F1-score
            min_roc_auc: Minimum acceptable ROC-AUC
            fairness_threshold: Maximum acceptable disparity in metrics across groups
        """
        self.thresholds = {
            'accuracy': min_accuracy,
            'precision': min_precision,
            'recall': min_recall,
            'f1': min_f1,
            'roc_auc': min_roc_auc
        }
        self.fairness_threshold = fairness_threshold
        self.validation_results = {}
    
    def validate_performance(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_proba: np.ndarray = None) -> Dict[str, bool]:
        """
        Validate model meets minimum performance thresholds
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
        
        Returns:
            Dictionary of metric checks (True = passed, False = failed)
        """
        print("\n" + "="*70)
        print("🎯 PERFORMANCE VALIDATION")
        print("="*70)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        checks = {}
        for metric_name, metric_value in metrics.items():
            threshold = self.thresholds[metric_name]
            passed = metric_value >= threshold
            checks[metric_name] = passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{metric_name.upper():12} = {metric_value:.4f} (Threshold: {threshold:.4f}) {status}")
        
        self.validation_results['performance'] = {
            'checks': checks,
            'metrics': metrics,
            'passed': all(checks.values())
        }
        
        return checks
    
    def validate_fairness(self,
                         X: pd.DataFrame,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         sensitive_features: List[str]) -> Dict[str, Dict]:
        """
        Validate fairness across demographic groups
        Checks if performance is consistent across sensitive features
        
        Args:
            X: Feature dataframe
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: List of column names to check for fairness
        
        Returns:
            Dictionary of fairness checks per sensitive feature
        """
        print("\n" + "="*70)
        print("⚖️  FAIRNESS VALIDATION")
        print("="*70)
        
        fairness_checks = {}
        
        for feature in sensitive_features:
            if feature not in X.columns:
                print(f"⚠️  Feature '{feature}' not found in data, skipping")
                continue
            
            print(f"\n📊 Analyzing: {feature}")
            
            groups = X[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = X[feature] == group
                if mask.sum() < 10:  # Skip groups with too few samples
                    continue
                
                group_acc = accuracy_score(y_true[mask], y_pred[mask])
                group_metrics[group] = group_acc
                print(f"   {group:20} Accuracy: {group_acc:.4f} (n={mask.sum()})")
            
            if len(group_metrics) >= 2:
                # Calculate disparity (max - min accuracy)
                max_acc = max(group_metrics.values())
                min_acc = min(group_metrics.values())
                disparity = max_acc - min_acc
                
                passed = disparity <= self.fairness_threshold
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"\n   Disparity: {disparity:.4f} (Threshold: {self.fairness_threshold:.4f}) {status}")
                
                fairness_checks[feature] = {
                    'group_metrics': group_metrics,
                    'disparity': disparity,
                    'passed': passed
                }
        
        all_passed = all(check['passed'] for check in fairness_checks.values())
        
        self.validation_results['fairness'] = {
            'checks': fairness_checks,
            'passed': all_passed
        }
        
        return fairness_checks
    
    def validate_robustness(self,
                           model,
                           X_test: pd.DataFrame,
                           y_test: np.ndarray,
                           noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, float]:
        """
        Validate model robustness to input perturbations
        Tests if model performance degrades gracefully with noise
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test labels
            noise_levels: Standard deviations of Gaussian noise to add
        
        Returns:
            Dictionary of accuracy at each noise level
        """
        print("\n" + "="*70)
        print("🛡️  ROBUSTNESS VALIDATION")
        print("="*70)
        
        # Baseline (no noise)
        baseline_pred = model.predict(X_test)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        
        print(f"\n📊 Baseline Accuracy (no noise): {baseline_acc:.4f}")
        
        robustness_results = {'baseline': baseline_acc}
        
        # Get numeric columns only
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        
        for noise_std in noise_levels:
            # Add Gaussian noise to numeric features
            X_noisy = X_test.copy()
            noise = np.random.normal(0, noise_std, size=(len(X_test), len(numeric_cols)))
            X_noisy[numeric_cols] = X_test[numeric_cols] + noise
            
            # Predict with noisy data
            noisy_pred = model.predict(X_noisy)
            noisy_acc = accuracy_score(y_test, noisy_pred)
            
            # Calculate degradation
            degradation = baseline_acc - noisy_acc
            degradation_pct = (degradation / baseline_acc) * 100
            
            robustness_results[f'noise_{noise_std}'] = noisy_acc
            
            # Check if degradation is acceptable (< 10%)
            passed = degradation_pct < 10
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"\n🔊 Noise Level: {noise_std:.2f}")
            print(f"   Accuracy: {noisy_acc:.4f}")
            print(f"   Degradation: {degradation:.4f} ({degradation_pct:.1f}%) {status}")
        
        self.validation_results['robustness'] = {
            'results': robustness_results,
            'passed': all((baseline_acc - acc) / baseline_acc < 0.1 
                         for acc in robustness_results.values() if acc != baseline_acc)
        }
        
        return robustness_results
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            save_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("📋 MODEL VALIDATION REPORT")
        report_lines.append("="*70)
        
        # Overall status
        all_passed = all(
            result.get('passed', False) 
            for result in self.validation_results.values()
        )
        
        overall_status = "✅ APPROVED FOR DEPLOYMENT" if all_passed else "❌ REJECTED - FIXES REQUIRED"
        report_lines.append(f"\n🎯 Overall Status: {overall_status}\n")
        
        # Performance validation
        if 'performance' in self.validation_results:
            report_lines.append("\n" + "="*70)
            report_lines.append("🎯 PERFORMANCE VALIDATION")
            report_lines.append("="*70)
            
            perf = self.validation_results['performance']
            for metric, value in perf['metrics'].items():
                passed = perf['checks'][metric]
                status = "✅" if passed else "❌"
                threshold = self.thresholds[metric]
                report_lines.append(
                    f"{status} {metric.upper():12} = {value:.4f} "
                    f"(Threshold: {threshold:.4f})"
                )
        
        # Fairness validation
        if 'fairness' in self.validation_results:
            report_lines.append("\n" + "="*70)
            report_lines.append("⚖️  FAIRNESS VALIDATION")
            report_lines.append("="*70)
            
            fair = self.validation_results['fairness']
            for feature, check in fair['checks'].items():
                status = "✅" if check['passed'] else "❌"
                report_lines.append(
                    f"\n{status} {feature.upper()}"
                )
                for group, metric in check['group_metrics'].items():
                    report_lines.append(f"   {group:20} Accuracy: {metric:.4f}")
                report_lines.append(
                    f"   Disparity: {check['disparity']:.4f} "
                    f"(Threshold: {self.fairness_threshold:.4f})"
                )
        
        # Robustness validation
        if 'robustness' in self.validation_results:
            report_lines.append("\n" + "="*70)
            report_lines.append("🛡️  ROBUSTNESS VALIDATION")
            report_lines.append("="*70)
            
            robust = self.validation_results['robustness']
            for noise_level, acc in robust['results'].items():
                report_lines.append(f"   {noise_level:20} Accuracy: {acc:.4f}")
        
        # Recommendations
        report_lines.append("\n" + "="*70)
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("="*70)
        
        if all_passed:
            report_lines.append("\n✅ Model meets all validation criteria")
            report_lines.append("✅ Ready for production deployment")
            report_lines.append("✅ Continue monitoring performance in production")
        else:
            report_lines.append("\n❌ Model validation failed. Actions required:")
            
            if not self.validation_results.get('performance', {}).get('passed', True):
                report_lines.append("   • Improve model performance (retrain, tune hyperparameters)")
            
            if not self.validation_results.get('fairness', {}).get('passed', True):
                report_lines.append("   • Address fairness issues (balance training data, adjust thresholds)")
            
            if not self.validation_results.get('robustness', {}).get('passed', True):
                report_lines.append("   • Improve model robustness (add noise during training, regularization)")
        
        report_lines.append("\n" + "="*70)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {save_path}")
        
        return report


def validate_model_from_mlflow(model_uri: str, 
                               data_path: str,
                               test_size: float = 0.2) -> Tuple[bool, str]:
    """
    Validate a model from MLflow registry
    
    Args:
        model_uri: MLflow model URI (e.g., 'models:/ChurnPredictor/1')
        data_path: Path to dataset CSV
        test_size: Fraction of data to use for testing
    
    Returns:
        Tuple of (validation_passed, report_text)
    """
    print(f"\n🔍 Loading model: {model_uri}")
    
    # Load model
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Load data
    print(f"📊 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Get predictions
    print("🔮 Generating predictions...")
    predictions = model.predict(X_test)
    
    # Handle probability predictions
    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
        y_proba = predictions[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
    else:
        y_pred = predictions
        y_proba = None
    
    # Initialize validator
    validator = ModelValidator(
        min_accuracy=0.75,
        min_precision=0.70,
        min_recall=0.65,
        min_f1=0.70,
        min_roc_auc=0.80,
        fairness_threshold=0.10
    )
    
    # Run validations
    validator.validate_performance(y_test, y_pred, y_proba)
    
    # Fairness check on sensitive features
    validator.validate_fairness(
        X_test, 
        y_test, 
        y_pred,
        sensitive_features=['gender', 'SeniorCitizen']
    )
    
    # Robustness check
    validator.validate_robustness(model, X_test, y_test)
    
    # Generate report
    report = validator.generate_report()
    
    # Determine if validation passed
    validation_passed = all(
        result.get('passed', False) 
        for result in validator.validation_results.values()
    )
    
    return validation_passed, report


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Example usage
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_PATH = PROJECT_ROOT / "Telco-Customer-Churn.csv"
    
    if len(sys.argv) > 1:
        model_uri = sys.argv[1]
    else:
        # Default to latest production model
        model_uri = "models:/ChurnPredictor/Production"
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║           🔍 AUTOMATED MODEL VALIDATION SYSTEM                 ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    if not DATA_PATH.exists():
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    try:
        passed, report = validate_model_from_mlflow(model_uri, str(DATA_PATH))
        
        print("\n" + report)
        
        if passed:
            print("\n🎉 Validation PASSED - Model approved for deployment!")
            sys.exit(0)
        else:
            print("\n❌ Validation FAILED - Model requires improvements before deployment")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
