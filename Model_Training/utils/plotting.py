import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pandas as pd


def plot_hf_training_history(
        log_history: List[Dict[str, Any]],
        output_dir: str,
        loss_plot_filename: str = "training_validation_loss.png",
        lr_plot_filename: str = "learning_rate.png",
        regression_metrics_plot_filename: str = "regression_metrics.png",
        classification_metrics_plot_filename: str = "classification_metrics.png"
):
    """
    Plots training and evaluation metrics from Hugging Face Trainer's log_history
    and saves them to files.

    Args:
        log_history (List[Dict[str, Any]]): The log_history from TrainerState.
                                            Each dict contains metrics like 'epoch', 'train_loss',
                                            'eval_loss', 'learning_rate', 'eval_mse', etc.
        output_dir (str): Directory where the plot images will be saved.
        loss_plot_filename (str): Filename for the loss plot.
        lr_plot_filename (str): Filename for the learning rate plot.
        regression_metrics_plot_filename (str): Filename for regression metrics plot.
        classification_metrics_plot_filename (str): Filename for classification metrics plot.
    """
    if not log_history:
        print("Plotting skipped: log_history is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style

    # Convert log_history to a pandas DataFrame for easier manipulation
    try:
        df = pd.DataFrame(log_history)
    except Exception as e:
        print(f"Error converting log_history to DataFrame: {e}. Plotting skipped.")
        return


    if 'train_loss' in df.columns:
        train_logs = df[df['train_loss'].notna()].copy()  # Entries with 'train_loss' are training steps
    else:
        print("Warning: 'train_loss' column not found in logs. Skipping training loss/LR plots if data is missing.")
        print("DataFrame columns:", df.columns)
        print("DataFrame head:\n", df.head().to_string())  # .to_string() for better console output of head
        print("Warning: 'train_loss' column not found in logs. Skipping training loss/LR plots if data is missing.")
        train_logs = pd.DataFrame() # Empty dataframe

    if not train_logs.empty:
        if 'epoch' not in train_logs.columns and 'step' in train_logs.columns:
            # If 'epoch' is missing for training steps, use 'step' as x-axis
            train_logs['x_axis_train'] = train_logs['step']
            train_x_label = 'Training Steps'
        elif 'epoch' in train_logs.columns:
            train_logs['x_axis_train'] = train_logs['epoch']
            train_x_label = 'Epochs (Training Steps)'
        else:  # Fallback if neither epoch nor step is present for training logs
            print(
                "Warning: Could not determine x-axis for training logs. Skipping training loss/LR plots if data is missing.")
            train_logs = pd.DataFrame()  # Empty dataframe
    else: # If train_logs started empty
        train_x_label = 'Training Steps/Epochs' # Default label

    # Evaluation logs - typically have 'eval_loss' and other 'eval_*' metrics
    if 'eval_loss' in df.columns:
        eval_logs = df[df['eval_loss'].notna()].copy()  # Entries with 'eval_loss' are evaluation steps
    else:
        print("Warning: 'eval_loss' column not found in logs. Skipping evaluation plots if data is missing.")
        eval_logs = pd.DataFrame()

    if not eval_logs.empty:
        if 'epoch' not in eval_logs.columns and 'step' in eval_logs.columns:
            eval_logs['x_axis_eval'] = eval_logs['step']
            eval_x_label = 'Evaluation Steps'
        elif 'epoch' in eval_logs.columns:
            eval_logs['x_axis_eval'] = eval_logs['epoch']
            eval_x_label = 'Epochs'
        else:  # Fallback
            print("Warning: Could not determine x-axis for evaluation logs. Skipping evaluation plots if data is missing.")
            eval_logs = pd.DataFrame()  # Empty dataframe
    else: # If eval_logs started empty
        eval_x_label = 'Evaluation Steps/Epochs' # Default label


    # --- 1. Plot Training and Validation Loss ---
    # MODIFIED: Changed 'loss' to 'train_loss' in conditions and plotting
    if not train_logs.empty and 'train_loss' in train_logs.columns and \
            not eval_logs.empty and 'eval_loss' in eval_logs.columns and \
            'x_axis_train' in train_logs.columns and 'x_axis_eval' in eval_logs.columns:
        try:
            plt.figure(figsize=(12, 6))

            # Plot training loss (potentially many points if logged per step)
            plt.plot(train_logs['x_axis_train'], train_logs['train_loss'], 'bo-', alpha=0.7, markersize=3, linewidth=1,
                     label='Training Loss (per step/epoch)')

            # Plot validation loss (typically per epoch)
            plt.plot(eval_logs['x_axis_eval'], eval_logs['eval_loss'], 'ro-', markersize=5,
                     label='Validation Loss (per eval)')

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
            if plt.get_fignums(): # Check if a figure is open before trying to close
                plt.close()
    else:
        print("Skipping loss plot: Missing 'train_loss' or 'eval_loss' or their x-axis data in logs.")

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
            if plt.get_fignums():
                plt.close()
    else:
        print("Skipping learning rate plot: Missing 'learning_rate' or its x-axis data in logs (Note: 'learning_rate' was not found in the provided DataFrame columns).")

    # --- 3. Plot Regression Metrics (MSE, MAE, R2 from evaluation logs) ---
    reg_metrics_to_plot = {'eval_mse': 'MSE', 'eval_mae': 'MAE', 'eval_r2': 'R² Score'}
    available_reg_metrics = {k: v for k, v in reg_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_reg_metrics and 'x_axis_eval' in eval_logs.columns:
        try:
            num_reg_metrics = len(available_reg_metrics)
            plt.figure(figsize=(7 * num_reg_metrics, 5))  # Adjust figure width based on number of metrics

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
            if plt.get_fignums():
                plt.close()
    else:
        print(
            "Skipping regression metrics plot: No evaluation regression metrics (eval_mse, eval_mae, eval_r2) or x-axis found in logs.")

    # --- 4. Plot Classification Metrics (Accuracies from evaluation logs) ---
    cls_metrics_to_plot = {
        'eval_cls_accuracy_from_logits': 'Cls Accuracy (Logits)',
        'eval_cls_accuracy_from_mapped_scores': 'Cls Accuracy (Mapped Scores)'
    }
    available_cls_metrics = {k: v for k, v in cls_metrics_to_plot.items() if k in eval_logs.columns}

    if not eval_logs.empty and available_cls_metrics and 'x_axis_eval' in eval_logs.columns:
        try:
            num_cls_metrics = len(available_cls_metrics)
            plt.figure(figsize=(7 * num_cls_metrics, 5))  # Adjust figure width

            plot_idx = 1
            for col_name, plot_label in available_cls_metrics.items():
                plt.subplot(1, num_cls_metrics, plot_idx)
                plt.plot(eval_logs['x_axis_eval'], eval_logs[col_name], 'co-', label=plot_label)
                plt.title(f'Validation {plot_label}')
                plt.xlabel(eval_x_label)
                plt.ylabel('Accuracy')
                plt.ylim(0, 1.05)  # Accuracy typically between 0 and 1
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
            if plt.get_fignums():
                plt.close()
    else:
        print(
            "Skipping classification metrics plot: No evaluation classification accuracy metrics or x-axis found in logs.")
