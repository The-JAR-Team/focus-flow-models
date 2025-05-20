import os
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Ensure numpy is imported
import seaborn as sns  # For prettier confusion matrices
from typing import List, Dict, Any, Optional


def plot_confusion_matrix(
        cm_data: List[List[int]],  # Expects a list of lists (JSON serializable)
        class_names: List[str],
        output_path: str,
        title: str = 'Confusion Matrix',
        cmap_name: str = 'Blues'  # Using standard matplotlib cmap name
):
    """
    Plots a confusion matrix using Seaborn and saves it to a file.

    Args:
        cm_data (List[List[int]]): The confusion matrix data.
        class_names (List[str]): List of class names for labels.
        output_path (str): Path to save the plot image.
        title (str): Title of the plot.
        cmap_name (str): Name of the colormap to use (e.g., 'Blues', 'Greens').
    """
    if not cm_data or not class_names:
        print(f"Skipping confusion matrix plot for '{title}': Empty data or class names.")
        return

    # Validate cm_data structure (list of lists of numbers)
    if not isinstance(cm_data, list) or not all(isinstance(row, list) for row in cm_data):
        print(
            f"Skipping confusion matrix plot for '{title}': cm_data is not a list of lists. Received type: {type(cm_data)}")
        return
    if cm_data and not all(isinstance(val, (int, float)) for row in cm_data for val in row):
        print(f"Skipping confusion matrix plot for '{title}': cm_data contains non-numeric values.")
        return

    # Ensure cm_data is not empty and dimensions match class_names if possible
    if len(cm_data) == 0 or len(cm_data[0]) == 0:
        print(f"Skipping confusion matrix plot for '{title}': cm_data contains empty lists.")
        return
    if len(cm_data) != len(class_names) or len(cm_data[0]) != len(class_names):
        print(
            f"Warning for '{title}': CM dimensions ({len(cm_data)}x{len(cm_data[0])}) do not match class_names length ({len(class_names)}). Plotting anyway.")

    try:
        cm_array = np.array(cm_data)  # Convert to numpy array for heatmap

        plt.figure(
            figsize=(max(6, int(len(class_names) * 0.8)), max(5, int(len(class_names) * 0.7))))  # Dynamic figure size
        sns.heatmap(cm_array, annot=True, fmt="d", cmap=cmap_name,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, linewidths=.5, annot_kws={"size": 10})
        plt.title(title, fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Adjust rotation and alignment
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout(pad=1.5)  # Add padding
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
        print(f"Confusion matrix '{title}' saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix '{title}': {e}")
        if plt.get_fignums():  # Check if a figure is currently open
            plt.close()


def plot_hf_training_history(
        log_history: List[Dict[str, Any]],
        output_dir: str,
        idx_to_name_map: Optional[Dict[int, str]] = None,  # For CM class names
        loss_plot_filename: str = "training_validation_loss.png",
        lr_plot_filename: str = "learning_rate.png",
        regression_metrics_plot_filename: str = "regression_metrics.png",
        classification_metrics_plot_filename: str = "classification_metrics.png",
        confusion_matrix_eval_filename: str = "confusion_matrix_eval.png",  # For eval set
        confusion_matrix_test_filename: str = "confusion_matrix_test.png"  # For test set
):
    """
    Plots training and evaluation metrics from Hugging Face Trainer's log_history.

    MODIFIED:
    - Added `idx_to_name_map` parameter.
    - Added `confusion_matrix_eval_filename` and `confusion_matrix_test_filename` parameters.
    - Calls `plot_confusion_matrix` for evaluation and test sets if data is available.
    """
    if not log_history:
        print("Plotting skipped: log_history is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')  # Using a seaborn style

    try:
        df = pd.DataFrame(log_history)
    except Exception as e:
        print(f"Error converting log_history to DataFrame: {e}. Plotting skipped.")
        return

    # --- Prepare data for plotting (extracting train and eval logs) ---
    # (Your existing logic for train_logs, eval_logs, train_x_label, eval_x_label should be here)
    # This part is crucial for the x-axes of the plots.
    # Simplified version for brevity, assuming 'epoch' or 'step' exists.
    if 'train_loss' in df.columns:
        train_logs = df[df['train_loss'].notna()].copy()
        train_logs['x_axis_train'] = train_logs.get('epoch', train_logs.get('step'))
        train_x_label = 'Epoch/Step'
    else:
        print("Warning: 'train_loss' not found. Training loss/LR plots might be skipped.")
        train_logs = pd.DataFrame()
        train_x_label = 'Epoch/Step'

    if 'eval_loss' in df.columns:
        eval_logs = df[df['eval_loss'].notna()].copy()
        eval_logs['x_axis_eval'] = eval_logs.get('epoch', eval_logs.get('step'))
        eval_x_label = 'Epoch/Step'
    else:
        print("Warning: 'eval_loss' not found. Evaluation plots might be skipped.")
        eval_logs = pd.DataFrame()
        eval_x_label = 'Epoch/Step'

    # --- 1. Plot Training and Validation Loss ---
    if not train_logs.empty and 'train_loss' in train_logs.columns and 'x_axis_train' in train_logs.columns and \
            not eval_logs.empty and 'eval_loss' in eval_logs.columns and 'x_axis_eval' in eval_logs.columns:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(train_logs['x_axis_train'], train_logs['train_loss'], 'bo-', alpha=0.7, markersize=3, linewidth=1,
                     label='Training Loss')
            plt.plot(eval_logs['x_axis_eval'], eval_logs['eval_loss'], 'ro-', markersize=5, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel(f'{train_x_label} / {eval_x_label}')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_path = os.path.join(output_dir, loss_plot_filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Loss plot saved to {save_path}")
        except Exception as e:
            print(f"Error plotting loss curves: {e}")
            if plt.get_fignums(): plt.close()
    else:
        print("Skipping loss plot: Missing 'train_loss' or 'eval_loss' or their x-axis data.")

    # --- 2. Plot Learning Rate ---
    if not train_logs.empty and 'learning_rate' in train_logs.columns and 'x_axis_train' in train_logs.columns:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(train_logs['x_axis_train'], train_logs['learning_rate'], 'go-', markersize=3, linewidth=1,
                     label='Learning Rate')
            plt.title('Learning Rate Over Time')
            plt.xlabel(train_x_label)
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_path = os.path.join(output_dir, lr_plot_filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Learning rate plot saved to {save_path}")
        except Exception as e:
            print(f"Error plotting learning rate: {e}")
            if plt.get_fignums(): plt.close()
    else:
        print("Skipping learning rate plot: Missing 'learning_rate' or its x-axis data.")

    # --- 3. Plot Regression Metrics (MSE, MAE, R2 from evaluation logs) ---
    reg_metrics_to_plot = {'eval_mse': 'MSE', 'eval_mae': 'MAE', 'eval_r2': 'R² Score'}
    available_reg_metrics = {k: v for k, v in reg_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_reg_metrics and 'x_axis_eval' in eval_logs.columns:
        try:
            num_reg_metrics = len(available_reg_metrics)
            plt.figure(figsize=(7 * num_reg_metrics, 5))
            plot_idx = 1
            for col_name, plot_label in available_reg_metrics.items():
                plt.subplot(1, num_reg_metrics, plot_idx)
                plt.plot(eval_logs['x_axis_eval'], eval_logs[col_name], 'mo-', label=plot_label)
                plt.title(f'Validation {plot_label}')
                plt.xlabel(eval_x_label)
                plt.ylabel(plot_label)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            plt.tight_layout()
            save_path = os.path.join(output_dir, regression_metrics_plot_filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Regression metrics plot saved to {save_path}")
        except Exception as e:
            print(f"Error plotting regression metrics: {e}")
            if plt.get_fignums(): plt.close()
    else:
        print("Skipping regression metrics plot: No relevant eval metrics or x-axis data.")

    # --- 4. Plot Classification Metrics (Accuracies from evaluation logs) ---
    cls_metrics_to_plot = {
        'eval_cls_accuracy_from_logits': 'Cls Accuracy (Logits)',
        'eval_cls_accuracy_from_mapped_scores': 'Cls Accuracy (Mapped Scores)'
    }
    available_cls_metrics = {k: v for k, v in cls_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_cls_metrics and 'x_axis_eval' in eval_logs.columns:
        try:
            num_cls_metrics = len(available_cls_metrics)
            plt.figure(figsize=(7 * num_cls_metrics, 5))
            plot_idx = 1
            for col_name, plot_label in available_cls_metrics.items():
                plt.subplot(1, num_cls_metrics, plot_idx)
                plt.plot(eval_logs['x_axis_eval'], eval_logs[col_name], 'co-', label=plot_label)
                plt.title(f'Validation {plot_label}')
                plt.xlabel(eval_x_label)
                plt.ylabel('Accuracy')
                plt.ylim(0, 1.05)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            plt.tight_layout()
            save_path = os.path.join(output_dir, classification_metrics_plot_filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Classification metrics plot saved to {save_path}")
        except Exception as e:
            print(f"Error plotting classification metrics: {e}")
            if plt.get_fignums(): plt.close()
    else:
        print("Skipping classification metrics plot: No relevant eval metrics or x-axis data.")

    # --- 5. Plot Confusion Matrices ---
    if idx_to_name_map is None:
        print("Skipping all confusion matrix plots: idx_to_name_map not provided.")
    else:
        # Class names sorted by index for CM labels
        class_names_sorted = [idx_to_name_map[k] for k in sorted(idx_to_name_map.keys())]

        # Helper to get CM data from the last relevant log entry
        def get_last_cm_data(df_logs: pd.DataFrame, cm_key: str) -> Optional[List[List[int]]]:
            if cm_key in df_logs.columns:
                # Filter out NaNs and get the last valid entry
                valid_cm_logs = df_logs[df_logs[cm_key].notna()]
                if not valid_cm_logs.empty:
                    cm_data = valid_cm_logs[cm_key].iloc[-1]
                    if isinstance(cm_data, list):  # Ensure it's list (JSON decoded from logs)
                        return cm_data
                    else:
                        print(f"Warning: Data for '{cm_key}' is not a list: {type(cm_data)}. Skipping CM plot.")
                        return None
            print(f"No data or column for '{cm_key}' found in logs. Skipping CM plot.")
            return None

        # Evaluation CM (from logits) - uses "eval_" prefix from Trainer
        eval_cm_logits_data = get_last_cm_data(df, "eval_confusion_matrix_logits")
        if eval_cm_logits_data:
            plot_confusion_matrix(
                cm_data=eval_cm_logits_data,
                class_names=class_names_sorted,
                output_path=os.path.join(output_dir, confusion_matrix_eval_filename + "_logits.png"),  # Add suffix
                title='Confusion Matrix (Eval - From Logits)'
            )

        # Evaluation CM (from mapped scores) - uses "eval_" prefix
        eval_cm_mapped_data = get_last_cm_data(df, "eval_confusion_matrix_mapped")
        if eval_cm_mapped_data:
            plot_confusion_matrix(
                cm_data=eval_cm_mapped_data,
                class_names=class_names_sorted,
                output_path=os.path.join(output_dir, confusion_matrix_eval_filename + "_mapped.png"),  # Add suffix
                title='Confusion Matrix (Eval - Mapped Scores)'
            )

        # Test CM (from logits) - uses "test_" prefix from Trainer (if test set was evaluated)
        test_cm_logits_data = get_last_cm_data(df, "test_confusion_matrix_logits")
        if test_cm_logits_data:
            plot_confusion_matrix(
                cm_data=test_cm_logits_data,
                class_names=class_names_sorted,
                output_path=os.path.join(output_dir, confusion_matrix_test_filename + "_logits.png"),  # Add suffix
                title='Confusion Matrix (Test - From Logits)'
            )

        # Test CM (from mapped scores) - uses "test_" prefix
        test_cm_mapped_data = get_last_cm_data(df, "test_confusion_matrix_mapped")
        if test_cm_mapped_data:
            plot_confusion_matrix(
                cm_data=test_cm_mapped_data,
                class_names=class_names_sorted,
                output_path=os.path.join(output_dir, confusion_matrix_test_filename + "_mapped.png"),  # Add suffix
                title='Confusion Matrix (Test - Mapped Scores)'
            )
