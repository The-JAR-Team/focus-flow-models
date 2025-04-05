import os
import pandas as pd
from Preprocess.Pipeline.DaiseeConfig import CACHE_DIR
from Preprocess.Pipeline.Encapsulation.SourceData import SourceData
from Preprocess.Pipeline.PipelineStage import PipelineStage


class SourceStage(PipelineStage):
    """
    A pipeline stage that loads the metadata CSV, splits it into Train, Test, and Validation subsets,
    and saves them to a subdirectory in CACHE_DIR. It returns a SourceData object that holds the CSV paths.
    """
    INTERNAL_VERSION = "01"

    def __init__(self, pipeline_version):
        self.pipeline_version = pipeline_version

    def _compose_version(self):
        return f"{self.INTERNAL_VERSION}_{self.pipeline_version}"

    def process(self, data=None, verbose=True):
        version_str = self._compose_version()
        metadata_path = os.path.join(os.path.dirname(__file__), "../MetaData/metadata.csv")
        source_dir = os.path.join(CACHE_DIR, f"Source {version_str}")
        os.makedirs(source_dir, exist_ok=True)
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

        if verbose:
            print("-------")
            print("Source stage")
            print(status_msg)
            print("Version:", version_str)
            print("passed!")
            print("-------")

        return SourceData(train_csv, test_csv, validation_csv)


# Example usage remains unchanged.
if __name__ == "__main__":
    pipeline_version = "01"
    source_stage = SourceStage(pipeline_version)
    source_data = source_stage.process(verbose=True)
