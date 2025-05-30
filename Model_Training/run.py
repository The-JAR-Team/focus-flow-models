# Model_Training/run.py
import sys
import os
import argparse
import importlib.util  # For dynamic module loading if we centralize it here

# Determine project_root from this file's location
# run.py is in Model_Training/, so project_root is its parent.
_PROJECT_ROOT_RUN_PY = os.path.abspath(os.path.dirname(__file__))  # This is Model_Training directory
_ACTUAL_PROJECT_ROOT_RUN_PY = os.path.dirname(_PROJECT_ROOT_RUN_PY)  # This is the parent of Model_Training

if _ACTUAL_PROJECT_ROOT_RUN_PY not in sys.path:
    sys.path.insert(0, _ACTUAL_PROJECT_ROOT_RUN_PY)
    # print(f"DEBUG: run.py added to sys.path: {_ACTUAL_PROJECT_ROOT_RUN_PY}")

# Now we can import from Model_Training.main
from Model_Training.main.run_training import main as actual_training_main, load_config_module


def main_entry(cmd_args_list=None):  # Expects a list of string args, like sys.argv[1:]
    """
    Sets up the Python path, parses arguments, loads config, and calls the main training function.
    """
    print("DEBUG: Executing Model_Training/run.py main_entry()")

    project_root = _ACTUAL_PROJECT_ROOT_RUN_PY  # Use the already determined project root

    parser = argparse.ArgumentParser(
        description="Run training with a specified configuration file via run.py.",
        # Allow --help to be passed from __main__.py
        # prog="python -m Model_Training" or "python Model_Training/run.py"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration Python file (e.g., configs/my_config.py or Model_Training/configs/my_config.py)"
    )

    # If cmd_args_list is None, parse_args uses sys.argv[1:] by default
    # If cmd_args_list is provided (e.g. from __main__.py), use that.
    args = parser.parse_args(args=cmd_args_list)

    # Resolve config_file_path:
    # 1. If absolute, use as is.
    # 2. If relative, assume it's relative to the project_root.
    if os.path.isabs(args.config):
        config_file_path_resolved = args.config
    else:
        config_file_path_resolved = os.path.join(project_root, args.config)

    if not os.path.exists(config_file_path_resolved):
        # Fallback: try relative to CWD if not found relative to project root
        config_file_path_cwd = os.path.abspath(args.config)
        if os.path.exists(config_file_path_cwd):
            config_file_path_resolved = config_file_path_cwd
            print(
                f"DEBUG: Config path '{args.config}' not found relative to project root, but found relative to CWD: {config_file_path_resolved}")
        else:
            print(
                f"Error: Configuration file not found at '{args.config}' (tried absolute, relative to project root '{project_root}', and relative to CWD).")
            sys.exit(1)

    try:
        exp_config_module_loaded = load_config_module(config_file_path_resolved)
    except Exception as e:
        print(f"Error loading configuration module from {config_file_path_resolved}: {e}")
        sys.exit(1)

    print("DEBUG: --- Attempting to start the actual training process via run.py ---")
    try:
        actual_training_main(exp_config_module=exp_config_module_loaded, project_root=project_root)
        print("DEBUG: --- Actual training process finished (called from run.py) ---")
    except Exception as e_train:
        print(f"ERROR during actual_training_main() called from run.py: {e_train}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print(f"DEBUG: Script '{os.path.abspath(__file__)}' executed as __main__.")
    # When run.py is executed directly, sys.argv[1:] will be used by parse_args by default.
    main_entry()
