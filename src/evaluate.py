"""
Model evaluation module for AI Traffic Accident Severity Prediction System.
Handles comprehensive evaluation, visualization, and reporting of model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import itertools


class ModelEvaluator:
    """Handles evaluation and visualization of trained models."""

    def __init__(self, class_names: List[str] = None):
        """
        Initialize the model evaluator.

        Args:
            class_names: List of class names for severity levels
        """
        self.class_names = class_names or ['Minor', 'Moderate', 'Serious', 'Severe']
        self.n_classes = len(self.class_names)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_confusion_matrix(self, cm: np.ndarray, title: str = 'Confusion Matrix',
                            normalize: bool = False, cmap: str = 'Blues',
                            save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix array
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            cmap: Colormap for the plot
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'{title} (Normalized)'
        else:
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set ticks
        tick_marks = np.arange(len(self.class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(self, y_test: np.ndarray, y_score: np.ndarray,
                      title: str = 'ROC Curve', save_path: str = None) -> plt.Figure:
        """
        Plot ROC curve for multiclass classification.

        Args:
            y_test: True labels
            y_score: Predicted probabilities
            title: Plot title
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=range(self.n_classes))
        n_classes = y_test_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot micro-average
        ax.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC (area = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)

        # Plot each class
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {self.class_names[i]} (area = {roc_auc[i]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        return fig

    def plot_precision_recall_curve(self, y_test: np.ndarray, y_score: np.ndarray,
                                   title: str = 'Precision-Recall Curve',
                                   save_path: str = None) -> plt.Figure:
        """
        Plot precision-recall curve for multiclass classification.

        Args:
            y_test: True labels
            y_score: Predicted probabilities
            title: Plot title
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score

        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=range(self.n_classes))
        n_classes = y_test_bin.shape[1]

        # Compute Precision-Recall and area for each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        # Compute micro-average
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_bin.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(y_test_bin.ravel(), y_score.ravel())

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot micro-average
        ax.plot(recall["micro"], precision["micro"],
                label=f'Micro-average PR (AP = {average_precision["micro"]:.2f})',
                color='gold', linestyle=':', linewidth=4)

        # Plot each class
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'PR curve of class {self.class_names[i]} (AP = {average_precision[i]:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")

        return fig

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              title: str = 'Feature Importance',
                              top_n: int = 20, save_path: str = None) -> plt.Figure:
        """
        Plot feature importance for tree-based models.

        Args:
            model: Trained model (should have feature_importances_ attribute)
            feature_names: List of feature names
            title: Plot title
            top_n: Number of top features to display
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(top_n), importances[indices])
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Feature Importance')
        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_performance_comparison(self, model_scores: Dict[str, Dict],
                                   metric: str = 'test_f1',
                                   title: str = 'Model Performance Comparison',
                                   save_path: str = None) -> plt.Figure:
        """
        Plot performance comparison across models.

        Args:
            model_scores: Dictionary of model scores
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        # Extract model names and scores
        model_names = []
        scores = []
        std_devs = []  # Placeholder for std dev if available

        for name, scores_dict in model_scores.items():
            if metric in scores_dict and scores_dict[metric] is not None:
                model_names.append(name)
                scores.append(scores_dict[metric])
                # Try to get std dev from cross-validation if available
                std_devs.append(scores_dict.get(f'{metric}_std', 0))

        if not model_names:
            raise ValueError(f"No models have valid scores for metric '{metric}'")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, scores, yerr=std_devs, capsize=5, alpha=0.8)
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to {save_path}")

        return fig

    def plot_class_distribution(self, y: np.ndarray, title: str = 'Class Distribution',
                              save_path: str = None) -> plt.Figure:
        """
        Plot distribution of classes.

        Args:
            y: Target array
            title: Plot title
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        # Count occurrences of each class
        unique, counts = np.unique(y, return_counts=True)

        # Map to class names if possible
        labels = [self.class_names[i] if i < len(self.class_names) else str(i) for i in unique]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, counts, alpha=0.8)
        ax.set_ylabel('Count')
        ax.set_title(title)
        fig.tight_layout()

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                   f'{count}', ha='center', va='bottom')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")

        return fig

    def generate_evaluation_report(self, model_scores: Dict[str, Dict],
                                 best_model_name: str,
                                 save_dir: str = "visualizations/") -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            model_scores: Dictionary of model scores
            best_model_name: Name of the best model
            save_dir: Directory to save visualizations

        Returns:
            Path to the generated report
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create visualizations
        print("Generating evaluation visualizations...")

        # Model performance comparison
        fig1 = self.plot_performance_comparison(
            model_scores, metric='test_f1',
            title='Model F1-Score Comparison',
            save_path=os.path.join(save_dir, 'model_comparison.png')
        )
        plt.close(fig1)

        fig2 = self.plot_performance_comparison(
            model_scores, metric='test_accuracy',
            title='Model Accuracy Comparison',
            save_path=os.path.join(save_dir, 'model_accuracy.png')
        )
        plt.close(fig2)

        print(f"Evaluation visualizations saved to {save_dir}")

        # Create a summary report
        report_path = os.path.join(save_dir, 'evaluation_summary.txt')
        with open(report_path, 'w') as f:
            f.write("AI Traffic Accident Severity Prediction - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Best Model: {best_model_name}\n\n")

            f.write("Model Performance Summary:\n")
            f.write("-" * 30 + "\n")
            for model_name, scores in model_scores.items():
                f.write(f"{model_name}:\n")
                for metric, value in scores.items():
                    if isinstance(value, (int, float)) and value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")

        print(f"Evaluation report saved to {report_path}")
        return report_path


def main():
    """Main function for testing the evaluator."""
    print("Model Evaluator Module - Testing with sample data")

    # Create sample evaluation data
    np.random.seed(42)
    n_samples = 200
    n_classes = 4

    # Generate sample predictions and true labels
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_score = np.random.rand(n_samples, n_classes)
    # Normalize scores to sum to 1 for each sample
    y_score = y_score / y_score.sum(axis=1, keepdims=True)

    # Generate sample confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # Initialize evaluator
    evaluator = ModelEvaluator(class_names=['Minor', 'Moderate', 'Serious', 'Severe'])

    # Test confusion matrix plot
    fig1 = evaluator.plot_confusion_matrix(
        cm, title='Confusion Matrix (Sample)',
        save_path='visualizations/confusion_matrix_sample.png'
    )
    plt.close(fig1)

    # Test ROC curve
    fig2 = evaluator.plot_roc_curve(
        y_true, y_score,
        title='ROC Curve (Sample)',
        save_path='visualizations/roc_curve_sample.png'
    )
    plt.close(fig2)

    # Test precision-recall curve
    fig3 = evaluator.plot_precision_recall_curve(
        y_true, y_score,
        title='Precision-Recall Curve (Sample)',
        save_path='visualizations/pr_curve_sample.png'
    )
    plt.close(fig3)

    # Test class distribution
    fig4 = evaluator.plot_class_distribution(
        y_true, title='Class Distribution (Sample)',
        save_path='visualizations/class_distribution_sample.png'
    )
    plt.close(fig4)

    print("Sample evaluations completed.")


if __name__ == "__main__":
    main()