class TensorStackingResult:
    """
    Encapsulation class for the tensor stacking stage.

    Attributes:
      tensor_stack (torch.Tensor): A tensor of shape
          (target_frames, num_landmarks, dims) where each missing frame is filled with -1.
      label (dict or int): The label information from the previous stage.
      clip_folder (str): The clip folder identifier.
    """

    def __init__(self, tensor_stack, label, clip_folder, subject_name=None, dataset_type=None):
        self.tensor_stack = tensor_stack
        self.label = label
        self.clip_folder = clip_folder
        self.dataset_type = dataset_type
        self.subject_name = subject_name
