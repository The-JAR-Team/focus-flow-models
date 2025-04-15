import os
import cv2
import glob
import numpy as np # Needed for NaN or other placeholders potentially
import pandas as pd

from Preprocess.Pipeline.Encapsulation.ExtractionResult import ExtractionResult # Assuming this exists
from Preprocess.Pipeline.PipelineStage import PipelineStage # Assuming this base class exists


class FrameExtractionStage(PipelineStage):
    """
    Extracts frames from video files at a desired FPS, optionally caching them.
    Handles different metadata label formats (e.g., DAiSEE numeric, EngageNet string)
    and outputs a standardized label dictionary.
    Requires dataset_root and cache_dir paths during initialization.
    """
    INTERNAL_VERSION = "02" # Incremented version due to logic change

    def __init__(self, pipeline_version: str, dataset_root: str, cache_dir: str,
                 save_frames: bool = True, desired_fps: float = 24.0,
                 jpeg_quality: int = 50, resize_width: int = None, resize_height: int = None):
        """
        Initializes the frame extraction stage. See class docstring for details.
        Args are the same as before.
        """
        if not os.path.isdir(dataset_root):
             print(f"Warning: Dataset root directory does not exist: {dataset_root}")
        if save_frames and not os.path.isdir(cache_dir):
             print(f"Warning: Cache directory does not exist: {cache_dir}. It will be created if needed.")

        self.pipeline_version = pipeline_version
        self.dataset_root = dataset_root
        self.cache_dir = cache_dir
        self.save_frames = save_frames
        self.desired_fps = desired_fps
        self.jpeg_quality = jpeg_quality
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.version_str = f"{self.INTERNAL_VERSION}_{self.pipeline_version}"

    def _standardize_label(self, row: dict) -> dict:
        """
        Creates a standardized label dictionary from the input row,
        handling different potential input label formats.
        """
        # Define the standard output structure
        standard_label = {
            "engagement_numeric": None, # Expects number (e.g., DAiSEE 0-3)
            "boredom_numeric": None,    # Expects number
            "confusion_numeric": None,  # Expects number
            "frustration_numeric": None,# Expects number
            "engagement_string": None   # Expects string (e.g., EngageNet 'Engaged')
            # Add other potential standard fields if needed
        }

        # --- Populate from DAiSEE-like columns ---
        daisee_keys = {
            "engagement": "engagement_numeric",
            "boredom": "boredom_numeric",
            "confusion": "confusion_numeric",
            "frustration": "frustration_numeric"
        }
        for input_key, output_key in daisee_keys.items():
            if input_key in row and pd.notna(row[input_key]): # Check if key exists and is not NaN/NaT
                try:
                    # Attempt to convert to float first (handles ints and floats), then potentially int
                    numeric_value = float(row[input_key])
                    # If it's conceptually an integer, store as int
                    if numeric_value == int(numeric_value):
                         standard_label[output_key] = int(numeric_value)
                    else:
                         standard_label[output_key] = numeric_value
                except (ValueError, TypeError):
                    # Handle cases where conversion fails if needed, or leave as None
                    print(f"Warning: Could not convert value '{row[input_key]}' for '{input_key}' to numeric. Leaving as None.")
                    pass # Keep the default None

        # --- Populate from EngageNet-like columns ---
        engagenet_key = "engagement_label" # Your column name from the compatible metadata
        if engagenet_key in row and pd.notna(row[engagenet_key]):
            standard_label["engagement_string"] = str(row[engagenet_key])

        return standard_label

    def process(self, row: dict, verbose: bool = False):
        """
        Processes a row from the metadata CSV to extract frames and standardized labels.

        Args:
            row (dict): A dictionary representing a row from the metadata CSV.
                        Expected keys: 'relative_path', 'subset', 'person', 'clip_folder',
                        and *either* DAiSEE label keys ('engagement', 'boredom', etc.)
                        *or* EngageNet label key ('engagement_label').
            verbose (bool): If True, print status messages. Defaults to False.

        Returns:
            ExtractionResult: An object containing extracted frames, paths, standardized labels, etc.

        Raises:
            RuntimeError: If the video file cannot be found or opened, or essential keys are missing.
        """
        # --- 1. Extract info from metadata row ---
        try:
            relative_path = row['relative_path']
            person_id = str(row['person']) # Expecting 'person' after metadata unification
            clip_id = str(row['clip_folder']) # Expecting 'clip_folder' after metadata unification
            dataset_type = str(row.get('subset', 'Unknown'))
        except KeyError as e:
             raise RuntimeError(f"Input row dictionary is missing expected key: {e}. Row: {row}")


        # --- 2. Standardize Labels ---
        label_dict = self._standardize_label(row)

        # --- 3. Determine video path and cache path ---
        video_path = os.path.join(self.dataset_root, relative_path)
        frames_dir = None

        if self.save_frames:
            frames_dir = os.path.join(
                self.cache_dir,
                f"FrameExtraction_{int(self.desired_fps)}_{self.jpeg_quality}",
                person_id,
                clip_id
            )
            os.makedirs(frames_dir, exist_ok=True)

            # --- 4. Check cache ---
            cached_frames_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if cached_frames_paths:
                try:
                    frames = [cv2.imread(fp) for fp in cached_frames_paths]
                    if all(f is not None for f in frames):
                         self._print_status(clip_id, len(frames), cached=True, verbose=verbose)
                         # Return standardized label_dict here too
                         return ExtractionResult(frames=frames, frames_dir=frames_dir, label=label_dict, clip_folder=clip_id, dataset_type=dataset_type)
                    else:
                         print(f"Warning: Failed to load some cached frames for {clip_id}. Re-extracting.")
                except Exception as e:
                    print(f"Warning: Error loading cached frames for {clip_id} ({e}). Re-extracting.")

        # --- 5. Find video file ---
        actual_video_file = self._find_video_file(video_path)
        if not actual_video_file or not os.path.exists(actual_video_file):
            raise RuntimeError(f"Video file not found for {clip_id} at expected path: {video_path} (derived from relative_path: {relative_path})")

        # --- 6. Extract frames ---
        frames = self._extract_frames(actual_video_file, frames_dir, f"{clip_id}_{self.version_str}")
        self._print_status(clip_id, len(frames), cached=False, verbose=verbose)

        # --- 7. Return Result ---
        return ExtractionResult(frames=frames, frames_dir=frames_dir if self.save_frames else None, label=label_dict, # Use standardized label_dict
                                clip_folder=clip_id, dataset_type=dataset_type, subject_name=person_id)

    # _print_status, _find_video_file, _extract_frames, _resize_frame methods remain the same as previous version
    # (Include them below for completeness)

    def _print_status(self, clip_id, num_frames, cached=False, verbose=True):
        """Helper to print status if verbose is True."""
        if verbose:
            status = "linked from cache" if cached else "extracted"
            print(f"  [FrameExtraction] {clip_id}: {num_frames} frames {status}.")

    def _find_video_file(self, folder_or_file: str) -> str or None:
        """Finds video file, accepting either a direct path or a folder."""
        if os.path.isfile(folder_or_file):
            return folder_or_file # Input is already a file path
        if not os.path.isdir(folder_or_file):
            return None # Input path doesn't exist or isn't a directory

        # Search for common video extensions within the directory
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"): # Consider making this configurable
            candidates = [f for f in os.listdir(folder_or_file) if f.lower().endswith(ext)]
            if candidates:
                return os.path.join(folder_or_file, candidates[0])
        return None

    def _extract_frames(self, video_file: str, output_dir: str or None, base_name: str) -> list:
        """Extracts frames from a video file at the desired FPS."""
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_file}")
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not (original_fps > 0): original_fps = 30.0

        time_between_frames = 1.0 / self.desired_fps if self.desired_fps > 0 else float('inf')
        next_capture_time = 0.0
        extracted_frames = []
        frame_index = 0
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality] # Consider adapting based on config (e.g., PNG)

        while True:
            ret, frame = cap.read()
            if not ret: break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time >= next_capture_time:
                if self.resize_width or self.resize_height:
                    frame = self._resize_frame(frame, self.resize_width, self.resize_height)

                if self.save_frames and output_dir:
                    # Consider making frame naming/extension configurable
                    frame_filename = f"{base_name}_frame_{frame_index:04d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, frame_filename), frame, imwrite_params)

                extracted_frames.append(frame)
                frame_index += 1
                next_capture_time += time_between_frames

        cap.release()
        return extracted_frames

    def _resize_frame(self, frame, target_width=None, target_height=None):
        """Resizes a frame maintaining aspect ratio if only one dimension is given."""
        if not target_width and not target_height: return frame
        (h, w) = frame.shape[:2]

        if target_width and target_height:
            dim = (target_width, target_height)
        elif target_width:
            ratio = target_width / float(w)
            dim = (target_width, int(h * ratio))
        else:
            ratio = target_height / float(h)
            dim = (int(w * ratio), target_height)

        interpolation = cv2.INTER_AREA if dim[0] * dim[1] < w * h else cv2.INTER_LINEAR # Consider making configurable
        return cv2.resize(frame, dim, interpolation=interpolation)
