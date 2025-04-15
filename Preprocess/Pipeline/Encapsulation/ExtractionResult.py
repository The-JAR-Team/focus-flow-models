class ExtractionResult:
    """
    Encapsulation class to hold the extraction result.

    Attributes:
      frames (list): List of extracted frames as NumPy arrays.
      frames_dir (str or None): Directory path where frames are saved, or None if not saved.
      label (dict or None): Dictionary containing label information.
      clip_folder (str or None): The clip folder identifier.
    """
    def __init__(self, frames, frames_dir=None, label=None, clip_folder=None, subject_name=None, dataset_type=None):
        self.frames = frames
        self.dataset_type = dataset_type
        self.frames_dir = frames_dir
        self.label = label
        self.clip_folder = clip_folder
        self.subject_name = subject_name
