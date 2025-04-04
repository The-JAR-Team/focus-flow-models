import os
import csv

########################
# User-adjustable settings
########################
DATASET_ROOT = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\DataSet"  # The folder with Train/, Test/, Validation/
LABEL_CSV_PATH = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\Labels\Labels.csv"
MALE_FILE_PATH = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\GenderClips\Males.txt"
FEMALE_FILE_PATH = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\GenderClips\Females.txt"
OUTPUT_CSV_PATH = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\metadata.csv"

DATASET_FOLDERS = ['Train', 'Test', 'Validation']
########################


def load_labels(label_csv_path):
    """
    Reads the label CSV with columns: ClipID,Boredom,Engagement,Confusion,Frustration
    Returns a dict: labels_dict[clip_id] = (boredom, engagement, confusion, frustration)
    where clip_id is something like "1100011002.avi".
    """
    labels_dict = {}
    with open(label_csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        print("CSV fieldnames as read by Python:", reader.fieldnames)
        for row in reader:
            clip_id = row['ClipID'].strip()  # e.g. "1100011002.avi"
            boredom = int(row['Boredom'])
            engagement = int(row['Engagement'])
            confusion = int(row['Confusion'])
            frustration = int(row['Frustration'])
            labels_dict[clip_id] = (boredom, engagement, confusion, frustration)
    return labels_dict


def load_clip_ids(gender_file_path):
    """
    Reads a text file that has one clip ID per line, e.g. "1100011002.avi".
    Returns a set of these IDs for quick membership checks.
    """
    s = set()
    with open(gender_file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            clip_id = line.strip()
            if clip_id:
                s.add(clip_id)
    return s


def main():
    # 1) Load labels and gender sets
    labels_dict = load_labels(LABEL_CSV_PATH)
    male_set = load_clip_ids(MALE_FILE_PATH)
    female_set = load_clip_ids(FEMALE_FILE_PATH)

    # We'll collect all subfolders that presumably contain exactly 1 video
    all_subfolders = []

    # 2) Gather subfolders from Train, Test, Validation
    for subset_name in DATASET_FOLDERS:
        subset_path = os.path.join(DATASET_ROOT, subset_name)
        if not os.path.isdir(subset_path):
            print(f"Skipping missing folder: {subset_path}")
            continue

        user_dirs = os.listdir(subset_path)
        for user_dir in user_dirs:
            user_path = os.path.join(subset_path, user_dir)
            if not os.path.isdir(user_path):
                continue

            subfolders = os.listdir(user_path)
            for clip_folder in subfolders:
                clip_folder_path = os.path.join(user_path, clip_folder)
                if os.path.isdir(clip_folder_path):
                    all_subfolders.append((subset_name, user_dir, clip_folder, clip_folder_path))

    total_subfolders = len(all_subfolders)
    if total_subfolders == 0:
        print("No subfolders found. Nothing to process.")
        return

    print(f"Found {total_subfolders} subfolders.\n")

    # We'll store the final rows for the CSV here
    rows = []

    # For progress updates every 5%
    next_progress_threshold = 5

    # 3) For each subfolder, figure out video, frames, gender, label, etc.
    for i, (subset_name, person_id, clip_folder, clip_folder_path) in enumerate(all_subfolders, start=1):
        # Progress check
        percent_done = (i / total_subfolders) * 100
        if percent_done >= next_progress_threshold:
            print(f"Progress: {percent_done:.1f}% ({i}/{total_subfolders} subfolders processed)")
            while percent_done >= next_progress_threshold:
                next_progress_threshold += 5

        # Check the directory contents
        contents = os.listdir(clip_folder_path)
        # Identify any video files
        video_files = [f for f in contents if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        # If not exactly one video, skip it
        if len(video_files) != 1:
            continue

        video_name = video_files[0]  # e.g. "1100011002.avi"
        clip_id = video_name  # for matching in the label/gender sets

        # ------------------------------------------------------------
        # Only proceed if this clip is in the labels_dict
        # If not, we skip it (i.e., no row is created).
        # ------------------------------------------------------------
        if clip_id not in labels_dict:
            continue

        boredom, engagement, confusion, frustration = labels_dict[clip_id]

        # Count frames: everything else in the folder is presumably frames
        num_frames = len(contents) - 1

        # Determine gender
        if clip_id in male_set:
            gender = "M"
        elif clip_id in female_set:
            gender = "F"
        else:
            gender = "U"  # unknown

        # Build relative path (ends at subfolder, not the video file)
        rel_path = os.path.relpath(clip_folder_path, DATASET_ROOT)

        # Create row
        row = {
            "subset": subset_name,
            "person": person_id,
            "clip_folder": clip_folder,
            "relative_path": rel_path,
            "gender": gender,
            "num_frames": num_frames,
            "boredom": boredom,
            "engagement": engagement,
            "confusion": confusion,
            "frustration": frustration
        }
        rows.append(row)

    # 4) Write out the final CSV
    fieldnames = [
        "subset",
        "person",
        "clip_folder",
        "relative_path",
        "gender",
        "num_frames",
        "boredom",
        "engagement",
        "confusion",
        "frustration"
    ]

    with open(OUTPUT_CSV_PATH, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone! Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}.")


if __name__ == "__main__":
    main()
