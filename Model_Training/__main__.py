# Model_Training/__main__.py
import sys
from Model_Training.run import main_entry


if __name__ == "__main__":
    # Call the main entry function from run.py, passing command line arguments
    # sys.argv[0] is the script name/module path, so pass args from index 1
    print(f"DEBUG: Executing Model_Training/__main__.py with args: {sys.argv[1:]}")
    main_entry(cmd_args_list=sys.argv[1:])
