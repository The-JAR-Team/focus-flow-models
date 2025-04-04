import os
import cv2

########################
# User-adjustable settings
########################
DATASET_ROOT = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\Daisee data\DataSet"  # <-- ABSOLUTE path to the folder containing Train/Test/Validation
DESIRED_FPS = 24.0
RESIZE_WIDTH = None     # e.g., 640 if you want to reduce frames to 640 px wide
RESIZE_HEIGHT = None    # e.g., 480 if you want to reduce frames to 480 px tall
JPEG_QUALITY = 50       # 0-100 (lower = more compression = smaller files)
DATASET_FOLDERS = ['Train', 'Test', 'Validation']
########################


def extract_frames_fixed_fps(video_path, output_dir, base_name,
                             desired_fps=24.0,
                             jpeg_quality=50,
                             resize_width=None,
                             resize_height=None):
    """
    Extract frames from the video at EXACTLY `desired_fps` frames per second.
    Saves frames as JPG with the specified quality and optional resizing.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_dir : str
        Where to save the frames.
    base_name : str
        Prefix for the output filenames (e.g., "videoName_frame_0000.jpg").
    desired_fps : float
        Desired frames per second to extract.
    jpeg_quality : int
        0-100, lower = higher compression (smaller file size).
    resize_width : int or None
        If set, resize frames to this width (maintaining aspect ratio if height is None).
    resize_height : int or None
        If set, resize frames to this height (maintaining aspect ratio if width is None).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 0

    # Attempt to read the original FPS, fallback if unknown
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0  # fallback

    # We'll advance time by 1/desired_fps seconds after saving a frame
    time_between_frames = 1.0 / desired_fps
    next_capture_time = 0.0

    # Setup JPEG compression parameters
    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames
            break

        # Current time in the video (seconds)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # If we've reached or passed our next capture time, save the frame
        if current_time >= next_capture_time:
            # Resize if requested
            if resize_width or resize_height:
                frame = resize_frame(frame, resize_width, resize_height)

            # Construct filename
            frame_filename = f"{base_name}_frame_{extracted_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame, imwrite_params)

            extracted_count += 1
            next_capture_time += time_between_frames

    cap.release()
    return extracted_count


def resize_frame(frame, target_width=None, target_height=None):
    """
    Resize the frame while maintaining aspect ratio if only one dimension is given.
    If both dimensions are given, resize exactly to that size.
    """
    (h, w) = frame.shape[:2]

    # Both width and height given -> exact resize
    if target_width and target_height:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Only width -> keep aspect ratio
    if target_width and not target_height:
        ratio = target_width / float(w)
        new_dim = (target_width, int(h * ratio))
        return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

    # Only height -> keep aspect ratio
    if target_height and not target_width:
        ratio = target_height / float(h)
        new_dim = (int(w * ratio), target_height)
        return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

    # No resizing
    return frame


def main():
    # Gather all videos from the dataset folders
    all_videos = []
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
                if not os.path.isdir(subfolder_path):
                    continue

                # We assume subfolder has one or more video files
                files_in_subfolder = os.listdir(subfolder_path)
                for file_name in files_in_subfolder:
                    if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(subfolder_path, file_name)
                        video_base_name = os.path.splitext(file_name)[0]
                        all_videos.append((video_path, subfolder_path, video_base_name))

    total_videos = len(all_videos)
    print(f"Found {total_videos} videos total.\n")

    # Process each video, printing progress every 10 videos + the last one
    for i, (video_path, output_dir, base_name) in enumerate(all_videos, start=1):
        # Print progress on multiples of 10 or for the last video
        if i % 10 == 0 or i == total_videos:
            percent = (i / total_videos) * 100
            print(f"[{i}/{total_videos} - {percent:.1f}%] Extracting at {DESIRED_FPS} FPS: {video_path}")

        extracted_count = extract_frames_fixed_fps(
            video_path=video_path,
            output_dir=output_dir,
            base_name=base_name,
            desired_fps=DESIRED_FPS,
            jpeg_quality=JPEG_QUALITY,
            resize_width=RESIZE_WIDTH,
            resize_height=RESIZE_HEIGHT
        )

        # Optionally print how many frames extracted (only on multiples of 10 or last)
        if i % 10 == 0 or i == total_videos:
            print(f"  -> Extracted {extracted_count} frames\n")


if __name__ == "__main__":
    main()
