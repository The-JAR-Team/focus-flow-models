import sys
import os
import Model_Training.main.run_training


def main_entry():
    """
    Sets up the Python path and calls the main training function.
    """
    print("DEBUG: Executing Model_Training/run.py main_entry()")

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_directory)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"DEBUG: Added project root to sys.path: {project_root}")

    print(f"DEBUG: Current sys.path includes: {sys.path[0]} (and others)") # Print first item for brevity


    # Call the main training function
    print("DEBUG: --- Attempting to start the actual training process ---")
    try:
        Model_Training.main.run_training.main()
        print("DEBUG: --- Actual training process finished ---")
    except Exception as e_train:
        print(f"ERROR during actual_training_main(): {e_train}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # This block executes when you run this script directly
    # (e.g., python Model_Training/run.py from the focus-flow-models directory,
    # or via your IDE pointing to this file).
    print(f"DEBUG: Script '{os.path.abspath(__file__)}' executed as __main__.")
    main_entry()