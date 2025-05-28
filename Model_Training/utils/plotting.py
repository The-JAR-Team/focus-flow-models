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
    (This function remains unchanged from your provided version,
     assuming it works correctly for confusion matrices.)
    """
    if not cm_data or not class_names:
        print(f"Skipping confusion matrix plot for '{title}': Empty data or class names.")
        return

    if not isinstance(cm_data, list) or not all(isinstance(row, list) for row in cm_data):
        print(
            f"Skipping confusion matrix plot for '{title}': cm_data is not a list of lists. Received type: {type(cm_data)}")
        return
    if cm_data and not all(isinstance(val, (int, float)) for row in cm_data for val in row): # Check elements
        # Check if all elements in all sublists are numbers
        is_numeric = True
        for row in cm_data:
            if not all(isinstance(val, (int, float)) for val in row):
                is_numeric = False
                break
        if not is_numeric:
            print(f"Skipping confusion matrix plot for '{title}': cm_data contains non-numeric values.")
            return

    if len(cm_data) == 0 or (len(cm_data) > 0 and len(cm_data[0]) == 0) : # check for empty list or list of empty lists
        print(f"Skipping confusion matrix plot for '{title}': cm_data contains empty lists.")
        return

    # Check if dimensions match, but plot anyway with a warning if they don't.
    # This allows plotting even if the CM from metrics has a different number of classes
    # than idx_to_name_map (e.g. if some classes were never predicted/true in a batch).
    if len(cm_data) != len(class_names) or (cm_data and len(cm_data[0]) != len(class_names)):
        print(
            f"Warning for '{title}': CM dimensions ({len(cm_data)}x{len(cm_data[0]) if cm_data else 0}) do not perfectly match class_names length ({len(class_names)}). Adapting class names for plot.")
        # Adapt class_names for plotting if dimensions mismatch
        # This is a simple adaptation; more sophisticated handling might be needed
        # depending on how sk_confusion_matrix (from metrics.py) handles labels.
        # If sk_confusion_matrix was given explicit labels matching num_classes_classification,
        # then cm_data should already have the correct dimensions.
        # This warning primarily guards against unexpected data.
        # For plotting, we'll use the provided class_names up to the dimensions of cm_data.
        # This part might need adjustment based on how `labels` is used in `sk_confusion_matrix`
        # in your `metrics.py`. If `sk_confusion_matrix` uses `labels=class_labels_for_cm`
        # (where class_labels_for_cm are sorted indices), then cm_data should always
        # have dimensions matching len(class_labels_for_cm).
        # The class_names passed here should correspond to those labels.

    try:
        cm_array = np.array(cm_data)

        # Adjust class_names if necessary for plotting based on cm_array dimensions
        # This ensures xticklabels/yticklabels match the heatmap dimensions
        plot_class_names_rows = class_names[:cm_array.shape[0]]
        plot_class_names_cols = class_names[:cm_array.shape[1]]


        plt.figure(
            figsize=(max(6, int(len(plot_class_names_cols) * 0.8)), max(5, int(len(plot_class_names_rows) * 0.7))))
        sns.heatmap(cm_array, annot=True, fmt="d", cmap=cmap_name,
                    xticklabels=plot_class_names_cols, yticklabels=plot_class_names_rows,
                    cbar=True, square=True, linewidths=.5, annot_kws={"size": 10})
        plt.title(title, fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix '{title}' saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix '{title}': {e}")
        if plt.get_fignums():
            plt.close()


def plot_hf_training_history(
        log_history: List[Dict[str, Any]],
        output_dir: str,
        idx_to_name_map: Optional[Dict[int, str]] = None,
        loss_plot_filename: str = "training_validation_loss.png",
        lr_plot_filename: str = "learning_rate.png",
        regression_metrics_plot_filename: str = "regression_metrics.png",
        classification_metrics_plot_filename: str = "classification_metrics.png",
        confusion_matrix_eval_filename: str = "confusion_matrix_eval.png",
        confusion_matrix_test_filename: str = "confusion_matrix_test.png"
):
    """
    Plots training and evaluation metrics from Hugging Face Trainer's log_history.
    Correctly identifies training loss, eval loss, and learning rate.
    """
    if not log_history:
        print("Plotting skipped: log_history is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    try:
        df = pd.DataFrame(log_history)
    except Exception as e:
        print(f"Error converting log_history to DataFrame: {e}. Plotting skipped.")
        return

    # --- Prepare data for plotting ---
    # Training logs: entries containing 'loss' and 'learning_rate' (logged during training steps)
    # Evaluation logs: entries containing 'eval_loss' (logged during evaluation steps)

    # Identify training log entries (typically have 'loss' and 'learning_rate')
    # Using .get() for x_axis to be robust if 'epoch' or 'step' is missing in some entries
    train_logs = df[df['learning_rate'].notna() & df['loss'].notna()].copy()
    if not train_logs.empty:
        train_logs['x_axis'] = train_logs.apply(lambda row: row.get('epoch', row.get('step')), axis=1)
        train_x_label = 'Epoch/Step' if 'epoch' in train_logs.columns or 'step' in train_logs.columns else 'Log Entry'
    else:
        print("Warning: No training logs found with 'learning_rate' and 'loss'. Training loss/LR plots might be skipped.")
        train_x_label = 'Epoch/Step' # Default

    # Identify evaluation log entries
    eval_logs = df[df['eval_loss'].notna()].copy()
    if not eval_logs.empty:
        eval_logs['x_axis'] = eval_logs.apply(lambda row: row.get('epoch', row.get('step')), axis=1)
        eval_x_label = 'Epoch/Step' if 'epoch' in eval_logs.columns or 'step' in eval_logs.columns else 'Log Entry'

        # Ensure x_axis for eval_logs is numeric and sorted for proper plotting
        eval_logs = eval_logs.dropna(subset=['x_axis'])
        eval_logs['x_axis'] = pd.to_numeric(eval_logs['x_axis'], errors='coerce')
        eval_logs = eval_logs.sort_values(by='x_axis')

    else:
        print("Warning: No evaluation logs found with 'eval_loss'. Evaluation plots might be skipped.")
        eval_x_label = 'Epoch/Step' # Default


    # --- 1. Plot Training and Validation Loss ---
    plot_loss = False
    if not train_logs.empty and 'loss' in train_logs.columns and 'x_axis' in train_logs.columns:
        plot_loss = True
    if not eval_logs.empty and 'eval_loss' in eval_logs.columns and 'x_axis' in eval_logs.columns:
        plot_loss = True

    if plot_loss:
        try:
            plt.figure(figsize=(12, 6))
            if not train_logs.empty and 'loss' in train_logs.columns and 'x_axis' in train_logs.columns:
                 # Ensure x_axis for train_logs is numeric and sorted
                temp_train_logs = train_logs.dropna(subset=['x_axis', 'loss'])
                temp_train_logs['x_axis'] = pd.to_numeric(temp_train_logs['x_axis'], errors='coerce')
                temp_train_logs = temp_train_logs.sort_values(by='x_axis')
                plt.plot(temp_train_logs['x_axis'], temp_train_logs['loss'], 'bo-', alpha=0.7, markersize=3, linewidth=1, label='Training Loss')

            if not eval_logs.empty and 'eval_loss' in eval_logs.columns and 'x_axis' in eval_logs.columns:
                plt.plot(eval_logs['x_axis'], eval_logs['eval_loss'], 'ro-', markersize=5, label='Validation Loss')

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
        print("Skipping loss plot: Missing 'loss' (for training) or 'eval_loss' (for validation) or their x-axis data in logs.")

    # --- 2. Plot Learning Rate ---
    if not train_logs.empty and 'learning_rate' in train_logs.columns and 'x_axis' in train_logs.columns:
        try:
            plt.figure(figsize=(10, 5))
            # Ensure x_axis for train_logs is numeric and sorted for LR plot
            temp_lr_logs = train_logs.dropna(subset=['x_axis', 'learning_rate'])
            temp_lr_logs['x_axis'] = pd.to_numeric(temp_lr_logs['x_axis'], errors='coerce')
            temp_lr_logs = temp_lr_logs.sort_values(by='x_axis')

            plt.plot(temp_lr_logs['x_axis'], temp_lr_logs['learning_rate'], 'go-', markersize=3, linewidth=1, label='Learning Rate')
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
        print("Skipping learning rate plot: Missing 'learning_rate' or its x-axis data in training logs.")

    # --- 3. Plot Regression Metrics (MSE, MAE, R2 from evaluation logs) ---
    reg_metrics_to_plot = {'eval_mse': 'MSE', 'eval_mae': 'MAE', 'eval_r2': 'R² Score'}
    # Filter for metrics actually present in eval_logs
    available_reg_metrics = {k: v for k, v in reg_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_reg_metrics and 'x_axis' in eval_logs.columns:
        try:
            num_reg_metrics = len(available_reg_metrics)
            plt.figure(figsize=(7 * num_reg_metrics, 5)) # Adjusted for potentially fewer metrics
            plot_idx = 1
            for col_name, plot_label in available_reg_metrics.items():
                plt.subplot(1, num_reg_metrics, plot_idx)
                # Ensure metric column is numeric
                metric_data = pd.to_numeric(eval_logs[col_name], errors='coerce')
                plt.plot(eval_logs['x_axis'], metric_data, 'mo-', label=plot_label)
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
        print("Skipping regression metrics plot: No relevant eval metrics or x-axis data in evaluation logs.")

    # --- 4. Plot Classification Metrics (Accuracies from evaluation logs) ---
    cls_metrics_to_plot = {
        'eval_cls_accuracy_from_logits': 'Cls Accuracy (Logits)',
        'eval_cls_accuracy_from_mapped_scores': 'Cls Accuracy (Mapped Scores)'
    }
    available_cls_metrics = {k: v for k, v in cls_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_cls_metrics and 'x_axis' in eval_logs.columns:
        try:
            num_cls_metrics = len(available_cls_metrics)
            plt.figure(figsize=(7 * num_cls_metrics, 5))
            plot_idx = 1
            for col_name, plot_label in available_cls_metrics.items():
                plt.subplot(1, num_cls_metrics, plot_idx)
                metric_data = pd.to_numeric(eval_logs[col_name], errors='coerce')
                plt.plot(eval_logs['x_axis'], metric_data, 'co-', label=plot_label)
                plt.title(f'Validation {plot_label}')
                plt.xlabel(eval_x_label)
                plt.ylabel('Accuracy')
                plt.ylim(0, 1.05) # Assuming accuracy is between 0 and 1
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
        print("Skipping classification metrics plot: No relevant eval metrics or x-axis data in evaluation logs.")

    # --- 5. Plot Confusion Matrices ---
    if idx_to_name_map is None:
        print("Skipping all confusion matrix plots: idx_to_name_map not provided.")
    else:
        class_names_sorted = [idx_to_name_map[k] for k in sorted(idx_to_name_map.keys())]

        def get_last_cm_data(df_logs: pd.DataFrame, cm_key: str) -> Optional[List[List[int]]]:
            if cm_key in df_logs.columns:
                valid_cm_logs = df_logs[df_logs[cm_key].notna()]
                if not valid_cm_logs.empty:
                    cm_data = valid_cm_logs[cm_key].iloc[-1]
                    if isinstance(cm_data, list):
                        # Basic check for list of lists structure
                        if all(isinstance(row, list) for row in cm_data):
                             # Further check if elements are numeric, handle potential errors
                            try:
                                # Attempt to convert to numpy array to catch type errors early
                                np.array(cm_data, dtype=float)
                                return cm_data
                            except ValueError:
                                print(f"Warning: Data for '{cm_key}' contains non-numeric values. Skipping CM plot.")
                                return None
                        else:
                            print(f"Warning: Data for '{cm_key}' is not a list of lists. Skipping CM plot.")
                            return None
                    else:
                        print(f"Warning: Data for '{cm_key}' is not a list: {type(cm_data)}. Skipping CM plot.")
                        return None
            # print(f"Info: No data or column for '{cm_key}' found in logs. Skipping CM plot for this key.") # Reduced verbosity
            return None

        # Evaluation CMs
        eval_cm_keys = {
            "eval_confusion_matrix_logits": "_logits.png",
            "eval_confusion_matrix_mapped": "_mapped.png"
        }
        for cm_key, suffix in eval_cm_keys.items():
            cm_data = get_last_cm_data(df, cm_key) # df instead of eval_logs, as CMs are logged once per eval
            if cm_data:
                plot_confusion_matrix(
                    cm_data=cm_data,
                    class_names=class_names_sorted,
                    output_path=os.path.join(output_dir, confusion_matrix_eval_filename + suffix),
                    title=f'Confusion Matrix (Eval - {cm_key.replace("eval_confusion_matrix_", "")})'
                )

        # Test CMs (if test evaluation was run)
        test_cm_keys = {
            "test_confusion_matrix_logits": "_logits.png",
            "test_confusion_matrix_mapped": "_mapped.png"
        }
        for cm_key, suffix in test_cm_keys.items():
            cm_data = get_last_cm_data(df, cm_key) # df instead of eval_logs
            if cm_data:
                plot_confusion_matrix(
                    cm_data=cm_data,
                    class_names=class_names_sorted,
                    output_path=os.path.join(output_dir, confusion_matrix_test_filename + suffix),
                    title=f'Confusion Matrix (Test - {cm_key.replace("test_confusion_matrix_", "")})'
                )
