# This file makes 'Model_Training' a Python package.

# Expose the main entry point function from Model_Training.run.py
# This allows running the training like:
# import Model_Training
# Model_Training.start_training()
# or
# from Model_Training import start_training
# start_training()

from .run import main_entry as start_training

# You can still expose other key components if desired:
# from .models.multitask_gru_attention_model_v4 import EngagementMultiTaskGRUAttentionModel
# from .utils.metrics import compute_metrics

# Define what symbols are exported when 'from Model_Training import *' is used.
# It's generally good practice to be explicit.
__all__ = [
    "start_training",
    # "EngagementMultiTaskGRUAttentionModel", # if you uncommented its import above
    # "compute_metrics", # if you uncommented its import above
]
