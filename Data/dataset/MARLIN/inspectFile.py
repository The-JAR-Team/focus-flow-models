import torch
import os

# --- Configuration ---
file_path = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\MARLIN\MARLIN_Test\subject_4_aoorj78dzj_vid_0_0.mp4.pt" # <-- CHANGE THIS to the actual path of one file

# --- Load the file ---
try:
    # Load the data. 'map_location=torch.device('cpu')' is useful if the file
    # was saved on a GPU machine and you're loading it on a CPU machine.
    loaded_data = torch.load(file_path, map_location=torch.device('cpu'))

    print(f"Successfully loaded: {file_path}")
    print("-" * 30)

    # --- Inspect the loaded data ---
    print(f"Type of loaded data: {type(loaded_data)}")
    print("-" * 30)

    # Common Scenario 1: It's a dictionary
    if isinstance(loaded_data, dict):
        print("Loaded data is a DICTIONARY. Keys:")
        print(list(loaded_data.keys()))
        print("-" * 30)

        # Try to print potential label information (adjust keys based on output above)
        possible_label_keys = ['label', 'labels', 'engagement', 'target', 'y']
        for key in possible_label_keys:
            if key in loaded_data:
                print(f"Found potential label key '{key}':")
                print(f"  Type: {type(loaded_data[key])}")
                # If it's a tensor, print its shape and maybe first few values
                if isinstance(loaded_data[key], torch.Tensor):
                     print(f"  Shape: {loaded_data[key].shape}")
                     print(f"  Example values: {loaded_data[key][:5]}") # Print first 5 values
                else:
                     print(f"  Value (or first few): {loaded_data[key][:5] if isinstance(loaded_data[key], (list, tuple)) else loaded_data[key]}")
                print("-" * 10)

        # You might also want to inspect other keys like 'features', 'video_id', 'subject_id' etc.
        # Example:
        # if 'features' in loaded_data and isinstance(loaded_data['features'], torch.Tensor):
        #    print(f"Features shape: {loaded_data['features'].shape}")


    # Common Scenario 2: It's a Tensor directly (less likely to contain labels AND features)
    elif isinstance(loaded_data, torch.Tensor):
        print("Loaded data is a TENSOR.")
        print(f"  Shape: {loaded_data.shape}")
        print(f"  Data type: {loaded_data.dtype}")
        print(f"  First few values: {loaded_data.flatten()[:10]}") # Print first 10 flattened values
        # In this case, the tensor might be *just* features or *just* labels, less likely both.

    # Common Scenario 3: It's a list or tuple
    elif isinstance(loaded_data, (list, tuple)):
         print(f"Loaded data is a {type(loaded_data).__name__} with {len(loaded_data)} elements.")
         print("Inspecting the first element:")
         first_element = loaded_data[0]
         print(f"  Type of first element: {type(first_element)}")
         if isinstance(first_element, torch.Tensor):
             print(f"  Shape of first element tensor: {first_element.shape}")
         else:
             print(f"  First element value (preview): {str(first_element)[:100]}...") # Preview
         # You'd need to figure out the structure - e.g., maybe it's [features, labels]?

    else:
        print("Loaded data is of an unexpected type. Further inspection needed.")
        print(loaded_data)


except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred while loading or inspecting the file: {e}")