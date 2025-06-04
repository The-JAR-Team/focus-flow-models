# Preprocess/Pipeline/Stages/FaceCloseUpStage.py

import cv2
import os
import numpy as np
from Preprocess.Pipeline.PipelineStage import PipelineStage  # Assuming this base class exists
from Preprocess.Pipeline.Encapsulation.ExtractionResult import ExtractionResult  # Assuming this is the correct path


class FaceCloseUpStage(PipelineStage):
    """
    A pipeline stage that detects faces in frames, crops them, and resizes them.
    Includes an option to display the middle processed frame for verification.
    """

    def __init__(self, output_width: int = 256, padding_factor: float = 0.2,
                 haar_cascade_path: str = None, display_middle_frame: bool = True):  # Added display_middle_frame
        """
        Initializes the face close-up stage.

        Args:
            output_width (int): The target width for the cropped and resized face image.
                                Height will be adjusted to maintain aspect ratio.
            padding_factor (float): Factor to add padding around the detected face bounding box.
                                    0.0 means no padding, 0.2 means 20% padding.
            haar_cascade_path (str): Path to the Haar Cascade XML file for face detection.
                                     If None, it will try to load a default one.
            display_middle_frame (bool): If True, displays the middle frame of each processed
                                         clip. Defaults to False.
        """
        self.output_width = output_width
        self.padding_factor = padding_factor
        self.display_middle_frame = display_middle_frame  # Store the new parameter

        if haar_cascade_path and os.path.exists(haar_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        else:
            # Try to load a common default path or raise an error
            # For this example, we'll point to where OpenCV often stores it.
            # Users might need to adjust this path or provide it explicitly.
            # You might need to install opencv-data-python: pip install opencv-data-python
            default_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(default_cascade_path):
                raise FileNotFoundError(
                    f"Haar Cascade file not found at specified path: {haar_cascade_path} "
                    f"or default path: {default_cascade_path}. "
                    "Please provide a valid 'haar_cascade_path' in the config "
                    "or ensure OpenCV's default cascades are accessible."
                )
            self.face_cascade = cv2.CascadeClassifier(default_cascade_path)

        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade for face detection.")

    def process(self, extraction_result: ExtractionResult, verbose: bool = False):
        """
        Processes frames to detect, crop, and resize faces. Optionally displays the middle frame.

        Args:
            extraction_result (ExtractionResult): The input data containing frames.
            verbose (bool): If True, print status messages.

        Returns:
            ExtractionResult: An object containing the processed (face-cropped) frames
                              and other original data.
        """
        frames = extraction_result.frames
        processed_frames = []
        # Use a default if clip_folder is None, for window titles and messages
        clip_identifier = extraction_result.clip_folder if extraction_result.clip_folder is not None else "UnknownClip"

        if not frames:
            if verbose:
                print(f"Warning: No frames received in FaceCloseUpStage for clip {clip_identifier}.")
            # Return the original extraction_result if there are no frames
            return extraction_result

        for idx, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                if verbose:
                    print(
                        f"Warning: Invalid frame encountered at index {idx} for clip {clip_identifier} in FaceCloseUpStage. Skipping.")
                processed_frames.append(frame)  # Append original invalid frame
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)  # Minimum face size to detect
            )

            if len(faces) == 0:
                if verbose:
                    print(
                        f"Warning: No face detected in frame {idx} for clip {clip_identifier}. Returning original frame.")
                processed_frames.append(frame)  # Return original frame if no face
                continue

            # If multiple faces, pick the largest one (by area)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Add padding
            pad_w = int(w * self.padding_factor)
            pad_h = int(h * self.padding_factor)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)

            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                if verbose:
                    print(
                        f"Warning: Cropped face is empty for frame {idx}, clip {clip_identifier}. Returning original frame.")
                processed_frames.append(frame)
                continue

            # Resize to output_width, maintaining aspect ratio
            original_h, original_w = cropped_face.shape[:2]
            if original_w == 0:  # Avoid division by zero
                if verbose:
                    print(
                        f"Warning: Cropped face has zero width for frame {idx}, clip {clip_identifier}. Returning original frame.")
                processed_frames.append(frame)
                continue

            aspect_ratio = original_h / original_w
            output_height = int(self.output_width * aspect_ratio)

            if output_height == 0:  # Ensure height is not zero
                output_height = 1  # Minimal height to avoid errors, or handle differently

            try:
                resized_face = cv2.resize(cropped_face, (self.output_width, output_height),
                                          interpolation=cv2.INTER_AREA)
                processed_frames.append(resized_face)
            except cv2.error as e:
                if verbose:
                    print(f"Error resizing frame {idx} for clip {clip_identifier}: {e}. Returning original frame.")
                processed_frames.append(frame)

        # Create a new ExtractionResult or modify the existing one
        extraction_result.frames = processed_frames

        # --- Display middle frame if enabled and frames exist ---
        if self.display_middle_frame and processed_frames:
            middle_frame_index = len(processed_frames) // 2
            middle_frame_to_display = processed_frames[middle_frame_index]

            if middle_frame_to_display is not None and middle_frame_to_display.size > 0:
                try:
                    window_title = f"Middle Frame: {clip_identifier} (Frame {middle_frame_index})"
                    cv2.imshow(window_title, middle_frame_to_display)
                    cv2.waitKey(0)  # Wait for a key press to close the window
                    cv2.destroyWindow(window_title)  # Destroy only this specific window to avoid conflicts
                except cv2.error as e:
                    # This error can occur if there's no GUI environment (e.g., running on a headless server)
                    print(f"OpenCV Error displaying middle frame for clip {clip_identifier}: {e}")
                    print("Note: Displaying images with cv2.imshow() requires a GUI environment.")
                    print("If running in a headless environment, this feature will not work.")
            elif verbose:  # Only print if verbose and display was attempted but frame was bad
                print(
                    f"Info: Middle frame for clip {clip_identifier} (index {middle_frame_index}) is invalid or empty, cannot display.")
        # --- End display logic ---

        if verbose:
            # A more accurate check for how many frames were actually changed by face cropping
            num_faces_actually_cropped = 0
            if len(frames) == len(processed_frames):  # Ensure lists are same length for zip
                for original_f, processed_f in zip(frames, processed_frames):
                    if not np.array_equal(original_f, processed_f):
                        num_faces_actually_cropped += 1

            print(
                f"  [FaceCloseUpStage] {clip_identifier}: Processed {len(processed_frames)} frames. Cropped/resized faces in {num_faces_actually_cropped} frames.")

        return extraction_result
