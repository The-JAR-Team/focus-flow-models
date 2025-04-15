# In Preprocess/Pipeline/Stages/MediapipeProcessingStage.py

import os
# --- Add threading import ---
import threading
# ---------------------------
import cv2
import mediapipe as mp
from Preprocess.Pipeline.PipelineStage import PipelineStage
# Make sure ExtractionResult is imported if needed by LandmarkExtractionResult or type hints
from Preprocess.Pipeline.Encapsulation.ExtractionResult import ExtractionResult
from Preprocess.Pipeline.Encapsulation.LandmarkExtractionResult import LandmarkExtractionResult

# --- Suppress TF/Mediapipe warnings here (keep this if it helps) ---
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --------------------------------------------------------------------


class MediapipeProcessingStage(PipelineStage):
    """
    A pipeline stage that processes frames using MediaPipe Face Mesh.
    Uses thread-local storage to ensure thread safety when run in parallel.
    """
    def __init__(self):
        """Initializes the stage, preparing thread-local storage."""
        self.mp_face_mesh = mp.solutions.face_mesh
        # --- Use threading.local() to store per-thread instances ---
        self.thread_local = threading.local()
        # --- Do NOT initialize self.face_mesh here ---

    def _get_thread_local_facemesh(self):
        """Gets or creates the FaceMesh instance for the current thread."""
        # Check if the current thread already has an instance
        if not hasattr(self.thread_local, 'face_mesh'):
            # If not, create a new one for this thread
            # print(f"Initializing FaceMesh for thread: {threading.current_thread().name}") # Optional debug print
            self.thread_local.face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                static_image_mode=False # Use False if processing video frames sequentially within a thread
            )
        return self.thread_local.face_mesh

    # Ensure type hint uses the correct input type
    def process(self, extraction_result: ExtractionResult, verbose=True):
        """Processes frames to extract landmarks using a thread-local FaceMesh instance."""
        frames = extraction_result.frames
        landmarks_list = []

        # --- Get the FaceMesh instance specific to this thread ---
        face_mesh = self._get_thread_local_facemesh()
        # --------------------------------------------------------

        if not frames:
             print("Warning: No frames received in MediapipeProcessingStage.")
             # Return result with empty landmarks but pass other data through
             return LandmarkExtractionResult(landmarks=[], label=extraction_result.label,
                                             frames_dir=extraction_result.frames_dir,
                                             clip_folder=extraction_result.clip_folder,
                                             dataset_type=extraction_result.dataset_type,
                                             subject_name=getattr(extraction_result, 'subject_name', None)) # Use getattr for safety

        for idx, frame in enumerate(frames):
            # Basic check for valid frame
            if frame is None or frame.size == 0:
                 # print(f"Warning: Invalid frame encountered at index {idx} for clip {extraction_result.clip_folder}. Skipping.")
                 landmarks_list.append(-1) # Append placeholder for invalid frame
                 continue

            try:
                # Convert the BGR image to RGB.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Process the image and find face landmarks.
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    # Assuming only one face, append its landmarks
                    landmarks_list.append(results.multi_face_landmarks[0])
                else:
                    # No landmarks detected for this frame
                    landmarks_list.append(-1) # Use -1 placeholder
            except Exception as frame_error:
                 # Catch potential errors during cvtColor or process
                 print(f"Error processing frame {idx} for clip {extraction_result.clip_folder}: {frame_error}")
                 landmarks_list.append(-1) # Append placeholder on error


        # --- Optional: Close the thread-local instance? ---
        # If FaceMesh has a close() method, call it here? Check Mediapipe docs.
        # Typically not needed unless managing GPU resources explicitly.
        # if hasattr(self.thread_local, 'face_mesh') and hasattr(self.thread_local.face_mesh, 'close'):
        #     self.thread_local.face_mesh.close()
        #     del self.thread_local.face_mesh # Clean up thread local attribute
        # --------------------------------------------------


        if verbose:
            total_frames = len(frames)
            failed = sum(1 for lm in landmarks_list if lm == -1)
            # Use tqdm.write if running within a tqdm loop in the main pipeline
            # print("-------")
            # print("MediapipeProcessing stage")
            # print(f"Clip: {extraction_result.clip_folder} - Processed {total_frames} frames. Failed: {failed}")
            # print("passed!")
            # print("-------")
            pass # Keep verbose printing minimal in threads

        # Ensure subject_name is passed through if it exists
        subject_name = getattr(extraction_result, 'subject_name', None)

        return LandmarkExtractionResult(landmarks=landmarks_list,
                                        label=extraction_result.label,
                                        frames_dir=extraction_result.frames_dir,
                                        clip_folder=extraction_result.clip_folder,
                                        dataset_type=extraction_result.dataset_type,
                                        subject_name=subject_name) # Pass subject_name

    def __del__(self):
        # Optional: Attempt to clean up FaceMesh instances when the stage object is deleted
        # Note: This might not reliably run for thread-local objects in all scenarios
        if hasattr(self, 'thread_local') and hasattr(self.thread_local, 'face_mesh'):
             if hasattr(self.thread_local.face_mesh, 'close'):
                  try:
                      self.thread_local.face_mesh.close()
                  except Exception as e:
                      print(f"Error closing thread-local FaceMesh: {e}")
             # Allow Python's garbage collection to handle the rest
