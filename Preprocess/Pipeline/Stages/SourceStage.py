# source_stage.py
import os
import pandas as pd
from Preprocess.Pipeline.DaiseeConfig import CACHE_DIR
from Preprocess.Pipeline.Encapsulation.SourceData import SourceData
from Preprocess.Pipeline.PipelineStage import PipelineStage


class SourceStage(PipelineStage):
    # Define internal version as a two-digit string
    INTERNAL_VERSION = "01"  # Update when internal changes occur

    def __init__(self, pipeline_version):
        """
        pipeline_version: A two-digit string (e.g., "01") representing the pipeline version.
        """
        self.pipeline_version = pipeline_version

    def _compose_version(self):
        """Combine the internal version with the pipeline version (e.g., "01_01")."""
        return f"{self.INTERNAL_VERSION}_{self.pipeline_version}"

    def process(self, data=None):
        """
        Loads metadata CSV, splits it, saves to a subdirectory in CACHE_DIR,
        and returns a SourceData object encapsulating the CSV paths.
        """
        # Define the metadata CSV location relative to this file
        metadata_path = os.path.join(os.path.dirname(__file__), "../MetaData/metadata.csv")
        version_str = self._compose_version()

        # Create a subdirectory for this source version
        source_dir = os.path.join(CACHE_DIR, f"Source {version_str}")
        os.makedirs(source_dir, exist_ok=True)

        # Prepare filenames with the version string in the new subdirectory
        train_csv = os.path.join(source_dir, f"Train_{version_str}.csv")
        test_csv = os.path.join(source_dir, f"Test_{version_str}.csv")
        validation_csv = os.path.join(source_dir, f"Val_{version_str}.csv")

        if os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(validation_csv):
            status_msg = "3 source files were found and linked"
        else:
            try:
                df = pd.read_csv(metadata_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load metadata.csv: {e}")

            train_df = df[df['subset'].str.lower() == 'train']
            test_df = df[df['subset'].str.lower() == 'test']
            validation_df = df[df['subset'].str.lower() == 'validation']

            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            validation_df.to_csv(validation_csv, index=False)
            status_msg = "3 source files were created"

        # Minimal status output
        print("-------")
        print("Source stage")
        print(status_msg)
        print("Version:", version_str)
        print("passed!")
        print("-------")

        return SourceData(train_csv, test_csv, validation_csv)


# Example usage (for testing this stage independently)
if __name__ == "__main__":
    pipeline_version = "01"
    source_stage = SourceStage(pipeline_version)
    source_data = source_stage.process()
