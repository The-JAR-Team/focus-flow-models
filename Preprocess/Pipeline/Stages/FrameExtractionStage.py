import os
import cv2
import glob
from Preprocess.Pipeline.Encapsulation.ExtractionResult import ExtractionResult
from Preprocess.Pipeline.PipelineStage import PipelineStage
from Preprocess.Pipeline.DaiseeConfig import DATASET_ROOT, CACHE_DIR


class FrameExtractionStage(PipelineStage):
    """
    A pipeline stage that extracts frames from a video at a fixed FPS and JPEG quality.
    It can extract frames directly into memory and optionally save them to a structured cache.
    It also extracts label information from the CSV row and pushes the clip_folder upstream.
    """
    INTERNAL_VERSION = "01"

    def __init__(self, pipeline_version: str, save_frames: bool = True, desired_fps: float = 24.0,
                 jpeg_quality: int = 50, resize_width: int = None, resize_height: int = None):
        self.pipeline_version = pipeline_version
        self.save_frames = save_frames
        self.desired_fps = desired_fps
        self.jpeg_quality = jpeg_quality
        self.resize_width = resize_width
        self.resize_height = resize_height

    def _compose_version(self):
        return f"{self.INTERNAL_VERSION}_{self.pipeline_version}"

    def process(self, row, verbose=True):
        version_str = self._compose_version()
        dataset_type = str(row['subset'])
        person_id = str(row['person'])
        clip_id = str(row['clip_folder'])
        relative_path = row['relative_path']
        label = {
            "engagement": row.get("engagement"),
            "boredom": row.get("boredom"),
            "confusion": row.get("confusion"),
            "frustration": row.get("frustration")
        }
        video_folder_or_file = os.path.join(DATASET_ROOT, relative_path)
        frames_dir = None
        if self.save_frames:
            frames_dir = os.path.join(
                CACHE_DIR,
                f"FrameExtraction{int(self.desired_fps)}_{self.jpeg_quality}",
                person_id,
                clip_id
            )
            os.makedirs(frames_dir, exist_ok=True)
            cached_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if cached_frames:
                frames = [cv2.imread(fp) for fp in cached_frames]
                self._print_status(clip_id, len(frames), version_str, cached=True, verbose=verbose)
                # Push clip_id as clip_folder.
                return ExtractionResult(frames=frames, frames_dir=frames_dir, label=label, clip_folder=clip_id)
        video_file = self._find_video_file(video_folder_or_file)
        if not video_file or not os.path.exists(video_file):
            raise RuntimeError(f"No valid video file found for row: {row}")
        frames = self._extract_frames(video_file, frames_dir, f"{clip_id}_{version_str}", self.save_frames, verbose=verbose)
        self._print_status(clip_id, len(frames), version_str, cached=False, verbose=verbose)
        return ExtractionResult(frames=frames, frames_dir=frames_dir if self.save_frames else None, label=label,
                                clip_folder=clip_id, dataset_type=dataset_type)

    def _print_status(self, clip_id, num_frames, version_str, cached=False, verbose=True):
        if verbose:
            print("-------")
            print("FrameExtraction stage")
            print(f"Clip folder: {clip_id}")
            print(f"{num_frames} frames {'found and linked (from cache)' if cached else 'extracted'}")
            print(f"Version: {version_str}")
            print("passed!")
            print("-------")

    def _find_video_file(self, folder_or_file: str) -> str:
        if os.path.isfile(folder_or_file):
            return folder_or_file
        if not os.path.isdir(folder_or_file):
            return None
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidates = glob.glob(os.path.join(folder_or_file, f"*{ext}"))
            if candidates:
                return candidates[0]
        return None

    def _extract_frames(self, video_file, output_dir, base_name, save_frames=True, verbose=True) -> list:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            if verbose:
                print(f"Could not open video: {video_file}")
            return []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0
        time_between_frames = 1.0 / self.desired_fps
        next_capture_time = 0.0
        extracted_frames = []
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time >= next_capture_time:
                if self.resize_width or self.resize_height:
                    frame = self._resize_frame(frame, self.resize_width, self.resize_height)
                if save_frames and output_dir is not None:
                    frame_filename = f"{base_name}_frame_{len(extracted_frames):04d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame, imwrite_params)
                extracted_frames.append(frame)
                next_capture_time += time_between_frames
        cap.release()
        return extracted_frames

    def _resize_frame(self, frame, target_width=None, target_height=None):
        (h, w) = frame.shape[:2]
        if target_width and target_height:
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if target_width and not target_height:
            ratio = target_width / float(w)
            new_dim = (target_width, int(h * ratio))
            return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        if target_height and not target_width:
            ratio = target_height / float(h)
            new_dim = (int(w * ratio), target_height)
            return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        return frame

# Example usage remains similar.
if __name__ == "__main__":
    row_data = {
        'subset': 'Train',
        'person': '110001',
        'clip_folder': '1100011002',
        'relative_path': 'Train\\110001\\1100011002',
        'gender': 'M',
        'num_frames': 0,
        'boredom': 0,
        'engagement': 2,
        'confusion': 0,
        'frustration': 0
    }
    stage = FrameExtractionStage(
        pipeline_version="01",
        save_frames=True,
        desired_fps=24.0,
        jpeg_quality=50
    )
    result = stage.process(row_data, verbose=True)
    print("FrameExtraction output:")
    print(f"Frames in memory: {len(result.frames)}")
    if result.frames_dir:
        print(f"Frames saved in directory: {result.frames_dir}")
    print("Extracted label:", result.label)
    print("Clip folder passed upstream:", result.clip_folder)
