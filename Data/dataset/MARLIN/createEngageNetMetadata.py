import os
import pandas as pd
from tqdm import tqdm # Optional: for a progress bar

########################
# User-adjustable settings
########################

# ---> Directory containing the dataset subsets with VIDEO FILES (e.g., MARLIN_Train, MARLIN_Val)
DATASET_ROOT = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\MARLIN" #<-- Your specific path

# ---> List of the dataset subset folders to scan within DATASET_ROOT
# Using 'Train', 'Validation' based on your edits
DATASET_FOLDERS = ['Train', 'Validation']

# ---> FULL paths to your TWO Excel label files
LABEL_XLSX_PATHS = [
    r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\MARLIN\train_engagement_labels.xlsx", #<-- Your specific path
    r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\MARLIN\validation_engagement_labels.xlsx" #<-- Your specific path
]

# ---> Path where the output metadata CSV file will be saved
OUTPUT_CSV_PATH = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\MARLIN\metadata_engagenet_daisee_compat.csv" # Renamed output for clarity

# ---> Column names expected in the Excel files
CHUNK_COLUMN_NAME = 'chunk' # Column with filenames like 'subject_7_...mp4'
LABEL_COLUMN_NAME = 'label' # Column with string labels like 'Barely-eng'

# ---> Video file extensions to look for
VIDEO_FILE_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

########################


def load_labels_from_xlsx(label_xlsx_paths, chunk_col, label_col):
    """
    Reads labels from one or more Excel files.
    Expects columns named by chunk_col and label_col.
    Returns a dict: labels_dict[chunk_id] = label_string
    where chunk_id is like "subject_7_as2uk9lhe2_vid_0_0.mp4".
    """
    all_labels_df = pd.DataFrame()
    print("Loading labels from Excel files...")
    for file_path in label_xlsx_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Label file not found, skipping: {file_path}")
            continue
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            print(f"  Read {len(df)} rows from {os.path.basename(file_path)}")
            if chunk_col not in df.columns or label_col not in df.columns:
                 print(f"  Error: Missing required columns ('{chunk_col}', '{label_col}') in {os.path.basename(file_path)}. Check column names.")
                 print(f"  Available columns: {df.columns.tolist()}")
                 continue
            df = df[[chunk_col, label_col]]
            if all_labels_df.empty:
                all_labels_df = df
            else:
                 all_labels_df = pd.concat([all_labels_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading Excel file {file_path}: {e}")

    if all_labels_df.empty:
        print("Error: No label data loaded. Exiting.")
        return {}

    all_labels_df.dropna(subset=[chunk_col, label_col], inplace=True)
    labels_dict = {}
    duplicates = 0
    for _, row in all_labels_df.iterrows():
        chunk_id = str(row[chunk_col]).strip()
        label_str = str(row[label_col]).strip()
        if chunk_id in labels_dict and labels_dict[chunk_id] != label_str:
            duplicates += 1
        labels_dict[chunk_id] = label_str

    if duplicates > 0:
         print(f"Warning: Found {duplicates} duplicate chunk IDs during label loading. The last encountered label was kept.")

    print(f"Successfully loaded labels for {len(labels_dict)} unique chunk IDs.")
    return labels_dict


def extract_person_id(filename):
    """
    Attempts to extract person/subject ID from filenames like 'subject_7_...'.
    Returns ID as string or None if not found. Changed name for clarity.
    """
    # Assumes filename format like subject_ID_rest...ext
    parts = filename.split('_')
    if len(parts) > 1 and parts[0] == 'subject':
        return parts[1]
    return None


def main():
    # 1) Load labels
    labels_dict = load_labels_from_xlsx(LABEL_XLSX_PATHS, CHUNK_COLUMN_NAME, LABEL_COLUMN_NAME)
    if not labels_dict:
        return

    print("-" * 30)
    print(f"Scanning dataset root for videos: {DATASET_ROOT}")
    print(f"Looking for folders: {DATASET_FOLDERS}")
    print(f"Looking for files ending with: {VIDEO_FILE_EXTENSIONS}")
    print("-" * 30)

    all_video_files = []

    # 2) Gather video file paths
    for subset_name in DATASET_FOLDERS:
        subset_path = os.path.join(DATASET_ROOT, subset_name)
        if not os.path.isdir(subset_path):
            print(f"Warning: Subset folder not found, skipping: {subset_path}")
            continue

        print(f"Scanning folder: {subset_path}...")
        found_count = 0
        # Assumes videos are directly in subset folder. Use os.walk if nested deeper.
        for filename in os.listdir(subset_path):
            if filename.lower().endswith(VIDEO_FILE_EXTENSIONS):
                file_path = os.path.join(subset_path, filename)
                # Use the actual folder name found (Train, Validation) as the subset identifier
                actual_subset_name = os.path.basename(os.path.dirname(file_path))
                all_video_files.append((actual_subset_name, filename, file_path))
                found_count += 1
        print(f"  Found {found_count} video files in {subset_name}.")


    total_files = len(all_video_files)
    if total_files == 0:
        print("\nNo video files found matching the criteria. Nothing to process.")
        return

    print(f"\nFound {total_files} potential video files. Matching with labels...")

    rows = []
    files_processed = 0
    files_matched = 0
    files_skipped_no_label = 0

    # 3) For each video file, find its label and create a metadata row
    for subset_name, video_filename, video_file_path in tqdm(all_video_files, desc="Processing video files"):
        files_processed += 1

        # The video filename is the key to look up in the labels dictionary
        chunk_id = video_filename

        if chunk_id in labels_dict:
            engagement_label = labels_dict[chunk_id]
            files_matched += 1

            # Extract 'person' ID (renamed from subject_id)
            person_id = extract_person_id(video_filename)
            # Create 'clip_folder' from filename without extension
            clip_folder_id = os.path.splitext(video_filename)[0]
            # Calculate relative path
            rel_path = os.path.relpath(video_file_path, DATASET_ROOT)

            # Create row dictionary with DAiSEE-like column names
            row = {
                "subset": subset_name,
                "person": person_id if person_id else 'Unknown', # Renamed from subject_id
                "clip_folder": clip_folder_id,                  # Added this field
                "relative_path": rel_path.replace("\\", "/"),   # Kept relative path
                "engagement_label": engagement_label          # Kept EngageNet label string
                # Omitted: video_filename, gender, num_frames, DAiSEE numeric labels
            }
            rows.append(row)
        else:
            files_skipped_no_label += 1


    print("-" * 30)
    print(f"Processing Summary:")
    print(f"  Total video files found: {total_files}")
    print(f"  Files processed: {files_processed}")
    print(f"  Files matched with labels: {files_matched}")
    print(f"  Files skipped (label not found): {files_skipped_no_label}")
    print("-" * 30)

    # 4) Write out the final CSV
    if not rows:
        print("No matching video files and labels found. Output CSV not created.")
        return

    # Define the order of columns in the output CSV to match desired structure
    fieldnames = [
        "subset",
        "person",           # Changed from subject_id
        "clip_folder",      # Added
        "relative_path",
        "engagement_label" # Kept EngageNet specific label
    ]

    try:
        output_df = pd.DataFrame(rows)
        output_df = output_df[fieldnames] # Ensure column order
        output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"Done! Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}.")
        print("\nOutput CSV Columns:", fieldnames) # Print final columns for confirmation
    except KeyError as e:
         print(f"\nError: A key listed in fieldnames was not found in the generated rows: {e}")
         print("Check the 'row' dictionary creation in step 3.")
    except Exception as e:
         print(f"\nError writing output CSV file: {e}")


if __name__ == "__main__":
    main()