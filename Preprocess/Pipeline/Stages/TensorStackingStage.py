import torch
import numpy as np
from Preprocess.Pipeline.PipelineStage import PipelineStage
from Preprocess.Pipeline.Encapsulation.TensorStackingResult import TensorStackingResult


class TensorStackingStage(PipelineStage):
    """
    A pipeline stage that converts the landmarks (from a LandmarkExtractionResult)
    into fixed-size tensor stacks. For each frame, if landmarks are missing (indicated by -1),
    a tensor of shape (num_landmarks, dims) filled with -1 is used.
    The resulting tensor stack is of shape (target_frames, num_landmarks, dims).
    It also pushes the clip_folder from the upstream result.
    """

    def __init__(self, target_frames=100, num_landmarks=478, dims=3):
        self.target_frames = target_frames
        self.num_landmarks = num_landmarks
        self.dims = dims

    def process(self, landmark_extraction_result, verbose=True):
        # Retrieve clip_folder from upstream result.
        landmarks_list = landmark_extraction_result.landmarks
        processed_frames = []
        num_missing = 0
        for idx, lm in enumerate(landmarks_list):
            if lm == -1:
                frame_tensor = torch.full((self.num_landmarks, self.dims), -1.0)
                num_missing += 1
            else:
                coords = []
                for landmark in lm.landmark:
                    coords.append([landmark.x, landmark.y, landmark.z])
                coords = np.array(coords)
                if coords.shape[0] < self.num_landmarks:
                    pad_size = self.num_landmarks - coords.shape[0]
                    padding = np.full((pad_size, self.dims), -1.0)
                    coords = np.vstack([coords, padding])
                elif coords.shape[0] > self.num_landmarks:
                    coords = coords[:self.num_landmarks, :]
                frame_tensor = torch.tensor(coords, dtype=torch.float32)
            processed_frames.append(frame_tensor)
        original_frames = len(processed_frames)
        if original_frames < self.target_frames:
            pad_frame = torch.full((self.num_landmarks, self.dims), -1.0)
            for _ in range(self.target_frames - original_frames):
                processed_frames.append(pad_frame.clone())
        elif original_frames > self.target_frames:
            processed_frames = processed_frames[:self.target_frames]
        tensor_stack = torch.stack(processed_frames)
        if verbose:
            print("-------")
            print("TensorStacking stage")
            print(f"Processed frames: {original_frames}")
            print(f"Missing landmarks replaced with -1: {num_missing}")
            if original_frames < self.target_frames:
                print(f"Padded with {self.target_frames - original_frames} extra frames")
            elif original_frames > self.target_frames:
                print(f"Truncated {original_frames - self.target_frames} frames")
            print(f"Final tensor stack shape: {tensor_stack.shape}")
            print(f"Label: {landmark_extraction_result.label}")
            print("passed!")
            print("-------")
        return TensorStackingResult(tensor_stack=tensor_stack,
                                    label=landmark_extraction_result.label,
                                    clip_folder=landmark_extraction_result.clip_folder,
                                    dataset_type=landmark_extraction_result.dataset_type)
