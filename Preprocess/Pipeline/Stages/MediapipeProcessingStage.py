import cv2
import mediapipe as mp
from Preprocess.Pipeline.PipelineStage import PipelineStage
from Preprocess.Pipeline.Encapsulation.LandmarkExtractionResult import LandmarkExtractionResult

class MediapipeProcessingStage(PipelineStage):
    """
    A pipeline stage that processes frames from an ExtractionResult using MediaPipe Face Mesh
    to extract face landmarks. If a frame fails to produce a face mesh, it returns -1 for that frame.
    The landmarks are kept in the raw format provided by MediaPipe.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    def process(self, extraction_result, verbose=True):
        frames = extraction_result.frames
        landmarks_list = []
        for idx, frame in enumerate(frames):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                landmarks_list.append(results.multi_face_landmarks[0])
            else:
                landmarks_list.append(-1)
        if verbose:
            total_frames = len(frames)
            failed = sum(1 for lm in landmarks_list if lm == -1)
            print("-------")
            print("MediapipeProcessing stage")
            print(f"Processed {total_frames} frames")
            print(f"Frames failed to detect landmarks: {failed}")
            print("passed!")
            print("-------")
        return LandmarkExtractionResult(landmarks=landmarks_list,
                                        label=extraction_result.label,
                                        frames_dir=extraction_result.frames_dir,
                                        clip_folder=extraction_result.clip_folder,
                                        dataset_type=extraction_result.dataset_type)
