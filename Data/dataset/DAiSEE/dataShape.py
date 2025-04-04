import os
import math

########################
# User-adjustable settings
########################
DATASET_ROOT = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\Daisee data\DataSet"  # <-- ABSOLUTE path to the folder containing Train/Test/Validation
DATASET_FOLDERS = ['Train', 'Test', 'Validation']
MIN_FRAMES = 240  # We'll warn if a folder has fewer frames than this
########################

def main():
    # 1) Gather all subfolders
    all_subfolders = []
    for folder_name in DATASET_FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        users = os.listdir(folder_path)
        for user_dir in users:
            user_path = os.path.join(folder_path, user_dir)
            if not os.path.isdir(user_path):
                continue

            video_subfolders = os.listdir(user_path)
            for subfolder in video_subfolders:
                subfolder_path = os.path.join(user_path, subfolder)
                if os.path.isdir(subfolder_path):
                    all_subfolders.append(subfolder_path)

    total_subfolders = len(all_subfolders)
    if total_subfolders == 0:
        print("No subfolders found. Nothing to check.")
        return

    print(f"Found {total_subfolders} subfolders.\n")

    # We'll keep:
    # 1) A dict of {frame_count: number_of_subfolders_with_that_count}
    # 2) A list of subfolders that are below the MIN_FRAMES threshold
    frame_counts_summary = {}
    below_min = []  # Will store tuples (video_path, num_frames)

    # For progress updates every 5%
    next_print_threshold = 5

    # 2) Process each subfolder
    for i, subfolder_path in enumerate(all_subfolders, start=1):
        # Progress check
        current_percent = (i / total_subfolders) * 100
        if current_percent >= next_print_threshold:
            print(f"Progress: {current_percent:.1f}% ({i}/{total_subfolders} subfolders checked)")
            while current_percent >= next_print_threshold:
                next_print_threshold += 5

        # List contents of this subfolder
        contents = os.listdir(subfolder_path)
        # Identify any video files
        video_files = [f for f in contents if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        # Skip if we don't have exactly one video file
        if len(video_files) != 1:
            continue

        # The video file name/path
        video_name = video_files[0]
        video_path = os.path.join(subfolder_path, video_name)

        # Everything else is presumably frames
        num_frames = len(contents) - 1

        # Update distribution
        frame_counts_summary[num_frames] = frame_counts_summary.get(num_frames, 0) + 1

        # Check if below minimum
        if num_frames < MIN_FRAMES:
            below_min.append((video_path, num_frames))

    # 3) Print final summary
    print("\n=== Distribution of Frame Counts ===")
    if not frame_counts_summary:
        print("No valid subfolders (none had exactly 1 video).")
    else:
        # Sort by frame_count
        for frame_count in sorted(frame_counts_summary.keys()):
            count_of_subfolders = frame_counts_summary[frame_count]
            print(f"{frame_count} frames: {count_of_subfolders} subfolder(s)")

    # 4) Print any below MIN_FRAMES
    if below_min:
        print(f"\n=== Subfolders Below {MIN_FRAMES} Frames ===")
        for vid_path, frames in below_min:
            print(f"{vid_path} has only {frames} frames.")
    else:
        print(f"\nNo subfolders below {MIN_FRAMES} frames.")

    print("\nDone checking extracted frames.")

if __name__ == '__main__':
    main()
