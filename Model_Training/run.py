import sys
import os


def main_entry():
    """
    Sets up the Python path and calls the main training function.
    """
    print("DEBUG: Executing Model_Training/run.py main_entry()")

    # Determine the project root directory.
    # __file__ is the path to this script (Model_Training/run.py).
    # os.path.dirname(__file__) is the directory of this script (Model_Training/).
    # The project root is the parent of the Model_Training/ directory.
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_directory)

    # Add the project root to sys.path if it's not already there.
    # This allows Python to find the 'Model_Training' package correctly
    # when importing modules like 'from Model_Training.main...'.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"DEBUG: Added project root to sys.path: {project_root}")

    print(f"DEBUG: Current sys.path includes: {sys.path[0]} (and others)") # Print first item for brevity

    # Now, import the main function from the training script
    try:
        # Since project_root (e.g., 'A:\JAR-team\focus-flow-models') is in sys.path,
        # we can use absolute-like imports starting from 'Model_Training'.
        from Model_Training.main.run_training import main as actual_training_main
        print("DEBUG: Successfully imported 'main' from Model_Training.main.run_training")
    except ImportError as e:
        print(f"CRITICAL ImportError: Could not import 'main' from Model_Training.main.run_training.")
        print(f"Error details: {e}")
        print("Please ensure the following:")
        print(f"  1. The file 'Model_Training/main/run_training.py' exists.")
        print(f"  2. An empty '__init__.py' file exists in 'Model_Training/' folder.")
        print(f"  3. An empty '__init__.py' file exists in 'Model_Training/main/' folder.")
        print(f"  4. The project root '{project_root}' was correctly added to sys.path.")
        return # Stop execution if import fails

    # Call the main training function
    print("DEBUG: --- Attempting to start the actual training process ---")
    try:
        actual_training_main()
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