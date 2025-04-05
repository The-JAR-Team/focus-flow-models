# LandmarkExtractionResult.py

class LandmarkExtractionResult:
    """
    Encapsulation class for the Mediapipe processing stage result.

    Attributes:
      landmarks (list): List of mediapipe landmarks objects (or None) for each frame.
      label (dict or None): Dictionary containing label information.
      frames_dir (str or None): Directory where frames were saved (if applicable).
    """

    def __init__(self, landmarks, label=None, frames_dir=None, clip_folder=None, dataset_type=None):
        self.landmarks = landmarks
        self.label = label
        self.frames_dir = frames_dir
        self.clip_folder = clip_folder
        self.dataset_type = dataset_type
