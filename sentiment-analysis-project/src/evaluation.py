"""
Evaluation Module for Twitter Sentiment Analysis

This module provides comprehensive evaluation metrics and analysis for sentiment models:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Classification Report
- ROC Curve and AUC
- Learning Curves
- Model Comparison
- Error Analysis

Following research standards for NLP model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import label_binarize
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for sentiment analysis.
    """
    
    def __init__(self, output_dir: str = "../results"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.confusion_matrices = {}
        self.classification_reports = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = "model",
                      class_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (Optional[np.ndarray]): Prediction probabilities
            model_name (str): Name of the model
            class_names (List[str]): Names of the classes
            
        Returns:
            Dict: Evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            if len(class_names) == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                try:
                    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
                    roc_auc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    logger.warning("Could not compute ROC AUC for multi-class")
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': dict(zip(class_names, precision_per_class)),
            'recall_per_class': dict(zip(class_names, recall_per_class)),
            'f1_per_class': dict(zip(class_names, f1_per_class)),
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc,
            'class_names': class_names
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        self.confusion_matrices[model_name] = cm
        self.classification_reports[model_name] = report
        
        logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            model_results (Dict[str, Dict]): Dictionary of model evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        logger.info("Comparing models...")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score']
            }
            
            if results['roc_auc'] is not None:
                row['ROC AUC'] = results['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        # Save comparison
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        logger.info("Model comparison saved")
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name (str): Name of the model
            figsize (Tuple[int, int]): Figure size
        """
        if model_name not in self.confusion_matrices:
            logger.error(f"No confusion matrix found for model {model_name}")
            return
        
        cm = self.confusion_matrices[model_name]
        results = self.evaluation_results[model_name]
        class_names = results['class_names']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix plot saved for {model_name}")
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred_proba: np.ndarray, 
                      model_name: str,
                      class_names: List[str] = None) -> None:
        """
        Plot ROC curve for binary or multi-class classification.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            class_names (List[str]): Names of the classes
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        plt.figure(figsize=(8, 6))
        
        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'roc_curve_{model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"ROC curve plot saved for {model_name}")
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray, 
                                   model_name: str,
                                   class_names: List[str] = None) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            class_names (List[str]): Names of the classes
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        plt.figure(figsize=(8, 6))
        
        if len(class_names) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})')
            
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            
            for i, class_name in enumerate(class_names):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'pr_curve_{model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Precision-Recall curve plot saved for {model_name}")
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison DataFrame
            figsize (Tuple[int, int]): Figure size
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        if 'ROC AUC' in comparison_df.columns:
            metrics.append('ROC AUC')
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.legend().remove()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Model comparison plot saved")
    
    def analyze_errors(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      texts: List[str],
                      model_name: str,
                      num_examples: int = 5) -> Dict[str, List[str]]:
        """
        Analyze prediction errors.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            texts (List[str]): Original texts
            model_name (str): Name of the model
            num_examples (int): Number of error examples to show
            
        Returns:
            Dict: Error analysis
        """
        logger.info(f"Analyzing errors for {model_name}")
        
        # Find misclassified examples
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(y_true),
            'false_positives': [],
            'false_negatives': [],
            'confusion_pairs': {}
        }
        
        # Get unique classes
        classes = np.unique(y_true)
        
        # Analyze confusion pairs
        for true_class in classes:
            for pred_class in classes:
                if true_class != pred_class:
                    indices = np.where((y_true == true_class) & (y_pred == pred_class))[0]
                    if len(indices) > 0:
                        error_analysis['confusion_pairs'][f'{true_class}->{pred_class}'] = {
                            'count': len(indices),
                            'examples': [texts[i] for i in indices[:num_examples]]
                        }
        
        # Save error analysis
        with open(os.path.join(self.output_dir, f'error_analysis_{model_name}.txt'), 'w') as f:
            f.write(f"Error Analysis for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Errors: {error_analysis['total_errors']}\n")
            f.write(f"Error Rate: {error_analysis['error_rate']:.4f}\n\n")
            
            f.write("Confusion Pairs:\n")
            f.write("-" * 30 + "\n")
            for pair, info in error_analysis['confusion_pairs'].items():
                f.write(f"{pair}: {info['count']} examples\n")
                for example in info['examples']:
                    f.write(f"  - {example}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved for {model_name}")
        
        return error_analysis
    
    def generate_evaluation_report(self, model_results: Dict[str, Dict]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_results (Dict[str, Dict]): Model evaluation results
            
        Returns:
            str: Evaluation report
        """
        logger.info("Generating evaluation report...")
        
        report = []
        report.append("Twitter Sentiment Analysis - Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison
        comparison_df = self.compare_models(model_results)
        report.append("Model Performance Comparison:")
        report.append("-" * 40)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best model
        best_model = comparison_df.iloc[0]['Model']
        report.append(f"Best Performing Model: {best_model}")
        report.append(f"Best F1 Score: {comparison_df.iloc[0]['F1 Score']:.4f}")
        report.append("")
        
        # Detailed results for each model
        for model_name, results in model_results.items():
            report.append(f"Detailed Results - {model_name}:")
            report.append("-" * 30)
            report.append(f"Accuracy: {results['accuracy']:.4f}")
            report.append(f"Precision: {results['precision']:.4f}")
            report.append(f"Recall: {results['recall']:.4f}")
            report.append(f"F1 Score: {results['f1_score']:.4f}")
            if results['roc_auc'] is not None:
                report.append(f"ROC AUC: {results['roc_auc']:.4f}")
            report.append("")
            
            report.append("Per-Class Performance:")
            for class_name in results['class_names']:
                report.append(f"  {class_name}:")
                report.append(f"    Precision: {results['precision_per_class'][class_name]:.4f}")
                report.append(f"    Recall: {results['recall_per_class'][class_name]:.4f}")
                report.append(f"    F1 Score: {results['f1_per_class'][class_name]:.4f}")
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(self.output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report_text)
        
        logger.info("Evaluation report generated and saved")
        
        return report_text
    
    def save_results(self) -> None:
        """
        Save all evaluation results to disk.
        """
        logger.info("Saving evaluation results...")
        
        # Save evaluation results
        with open(os.path.join(self.output_dir, 'evaluation_results.pkl'), 'wb') as f:
            import pickle
            pickle.dump(self.evaluation_results, f)
        
        # Save confusion matrices
        with open(os.path.join(self.output_dir, 'confusion_matrices.pkl'), 'wb') as f:
            import pickle
            pickle.dump(self.confusion_matrices, f)
        
        # Save classification reports
        with open(os.path.join(self.output_dir, 'classification_reports.pkl'), 'wb') as f:
            import pickle
            pickle.dump(self.classification_reports, f)
        
        logger.info("All evaluation results saved")


def main():
    """
    Example usage of the ModelEvaluator.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.choice([0, 1, 2], size=n_samples)
    y_pred = np.random.choice([0, 1, 2], size=n_samples)
    y_pred_proba = np.random.rand(n_samples, 3)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize
    
    sample_texts = [f"Sample text {i}" for i in range(n_samples)]
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    results = evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, 
        model_name="sample_model",
        class_names=["Negative", "Neutral", "Positive"]
    )
    
    # Generate comparison
    comparison_df = evaluator.compare_models({"sample_model": results})
    print("Model Comparison:")
    print(comparison_df)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix("sample_model")
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(y_true, y_pred, sample_texts, "sample_model")
    print(f"Error rate: {error_analysis['error_rate']:.4f}")
    
    # Generate report
    report = evaluator.generate_evaluation_report({"sample_model": results})
    print("Evaluation report generated successfully")


if __name__ == "__main__":
    main()
