import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For random splitting
from Preprocess.Pipeline.Encapsulation.SourceData import SourceData # Assuming this exists
from Preprocess.Pipeline.PipelineStage import PipelineStage # Assuming this base class exists


class SourceStage(PipelineStage):
    """
    A pipeline stage that loads a specified metadata CSV and prepares
    Train, Validation, and Test subsets.
    It can operate in two modes:
    1. 'column': Splits based on an existing 'subset' column in the CSV.
    2. 'random': Performs a stratified random split (if label available)
                 or simple random split of the entire dataset according
                 to specified ratios, ignoring any existing 'subset' column.
    Saves the resulting splits to CSV files in a subdirectory within cache_dir.
    Returns a SourceData object holding paths to these split CSV files.
    """
    INTERNAL_VERSION = "02" # Keep version as split logic is refined

    def __init__(self,
                 pipeline_version: str,
                 cache_dir: str,
                 metadata_csv_path: str,
                 perform_random_split: bool = False,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 random_seed: int = 42,
                 stratify_column: str = None):
        """ Initializes the source stage. See class docstring for details. """
        self.pipeline_version = pipeline_version
        self.cache_dir = cache_dir
        self.metadata_csv_path = metadata_csv_path
        self.perform_random_split = perform_random_split
        self.random_seed = random_seed
        self.stratify_column = stratify_column

        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Train ({train_ratio}), Validation ({val_ratio}), and Test ({test_ratio}) ratios must sum to 1.0. Current sum: {total_ratio}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if not os.path.isdir(cache_dir):
             print(f"Warning: Cache directory does not exist: {cache_dir}. It will be created.")
        if not os.path.isfile(metadata_csv_path):
             print(f"Warning: Input metadata CSV file not found at: {metadata_csv_path}")

    def _compose_version(self):
        """ Combines internal and pipeline versions for unique naming. """
        split_mode_str = "random" if self.perform_random_split else "column"
        seed_str = f"_seed{self.random_seed}" if self.perform_random_split else ""
        strat_str = f"_stratified-{self.stratify_column}" if self.perform_random_split and self.stratify_column else ""
        # Use a simplified identifier if not stratifying
        if self.perform_random_split and not self.stratify_column:
             strat_str = "_simple"
        return f"{self.INTERNAL_VERSION}_{self.pipeline_version}_{split_mode_str}{seed_str}{strat_str}"

    def process(self, data=None, verbose=True):
        """ Loads, splits (by column or randomly), and saves metadata CSV subsets. """
        version_str = self._compose_version()
        source_dir = os.path.join(self.cache_dir, f"Source_{version_str}")
        os.makedirs(source_dir, exist_ok=True)

        train_csv = os.path.join(source_dir, f"Train_{version_str}.csv")
        test_csv = os.path.join(source_dir, f"Test_{version_str}.csv")
        validation_csv = os.path.join(source_dir, f"Val_{version_str}.csv") # Sticking to 'Val' for consistency

        if os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(validation_csv):
            status_msg = f"Source files found and linked (from cache - split: {'random' if self.perform_random_split else 'column'})"
            files_created = False
        else:
            files_created = True
            if verbose:
                print(f"  [SourceStage] Cache miss for split mode '{'random' if self.perform_random_split else 'column'}'.")
                print(f"  [SourceStage] Loading main metadata from: {self.metadata_csv_path}")
            try:
                df = pd.read_csv(self.metadata_csv_path)
                if verbose: print(f"  [SourceStage] Total rows loaded from metadata: {len(df)}")
                if len(df) == 0: raise ValueError("Input metadata CSV is empty.")

                # --- Perform Split ---
                if self.perform_random_split:
                    status_msg = "Source files created (random split)"
                    print(f"  [SourceStage] Performing random split: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}, Seed={self.random_seed}")

                    # --- Stratification Setup ---
                    stratify_array = None
                    can_stratify = False
                    if self.stratify_column:
                        if self.stratify_column not in df.columns:
                            print(f"Warning: Stratify column '{self.stratify_column}' not found in metadata. Performing simple random split.")
                        else:
                            stratify_array = df[self.stratify_column]
                            value_counts = stratify_array.value_counts()
                            print(f"  [SourceStage] Attempting stratification on column: '{self.stratify_column}'")
                            if verbose: print(f"  [SourceStage] Stratify column value counts (original):\n{value_counts}")
                            # Check if *all* classes have at least 2 samples
                            if (value_counts >= 2).all():
                                can_stratify = True
                                print("  [SourceStage] Stratification possible (all classes have >= 2 samples).")
                            else:
                                print(f"Warning: Stratify column '{self.stratify_column}' contains classes with < 2 samples. Stratification disabled. Performing simple random split.")
                                stratify_array = None # Disable stratification
                    else:
                         print("  [SourceStage] Performing simple random split (no stratification column specified).")


                    try:
                        # --- 1. Split off Test set ---
                        # If df is very small, train_test_split might still yield empty sets
                        if len(df) < 2 : # Need at least 2 samples to split
                            print("Warning: Dataset too small to perform Train/Test split reliably.")
                            remaining_df, test_df = df, pd.DataFrame(columns=df.columns) # Put all in remaining
                        else:
                             remaining_df, test_df = train_test_split(
                                 df,
                                 test_size=self.test_ratio if len(df) * self.test_ratio >= 1 else 0, # Ensure test_size >= 1 sample if possible
                                 random_state=self.random_seed,
                                 stratify=stratify_array if can_stratify else None # Only stratify if possible
                             )
                        if verbose: print(f"  [SourceStage] After 1st split (Test): Remaining={len(remaining_df)}, Test={len(test_df)}")

                        # --- 2. Split Remainder into Train and Validation ---
                        stratify_array_remain = None
                        can_stratify_remain = False
                        if can_stratify and not remaining_df.empty:
                            stratify_array_remain = stratify_array.loc[remaining_df.index]
                            remain_value_counts = stratify_array_remain.value_counts()
                            # Check if stratification is still possible on the remainder
                            if (remain_value_counts >= 2).all():
                                can_stratify_remain = True
                            else:
                                print(f"Warning: Cannot stratify Train/Val split (some classes in remainder have < 2 samples). Performing simple random split for Train/Val.")
                                # stratify_array_remain = None # Keep variable for logic below, but won't be used if can_stratify_remain is False


                        # Calculate validation proportion relative to the *remaining* data
                        # Avoid division by zero if train+val ratio is 0
                        denominator = self.train_ratio + self.val_ratio
                        val_proportion_of_remainder = self.val_ratio / denominator if denominator > 0 else 0

                        # Handle edge cases for the second split
                        if remaining_df.empty:
                            train_df = pd.DataFrame(columns=df.columns)
                            validation_df = pd.DataFrame(columns=df.columns)
                        elif len(remaining_df) < 2 or np.isclose(val_proportion_of_remainder, 0): # Cannot split further or no val set needed
                            train_df = remaining_df
                            validation_df = pd.DataFrame(columns=df.columns)
                        elif np.isclose(val_proportion_of_remainder, 1): # All remaining goes to validation
                            train_df = pd.DataFrame(columns=df.columns)
                            validation_df = remaining_df
                        else:
                             # Ensure test_size for validation is at least 1 sample if possible and ratio > 0
                            val_test_size = val_proportion_of_remainder if len(remaining_df) * val_proportion_of_remainder >= 1 else 0
                            if np.isclose(val_test_size, 0) and self.val_ratio > 0: # If val ratio > 0, try to get at least 1 sample
                                 val_test_size = 1 / len(remaining_df) if len(remaining_df) > 0 else 0

                            train_df, validation_df = train_test_split(
                                remaining_df,
                                test_size=val_test_size,
                                random_state=self.random_seed, # Use same seed
                                stratify=stratify_array_remain if can_stratify_remain else None # Stratify only if possible
                            )

                        if verbose: print(f"  [SourceStage] After 2nd split (Train/Val): Train={len(train_df)}, Val={len(validation_df)}")

                        # Add final check and warning for empty validation set
                        if len(validation_df) == 0 and self.val_ratio > 0 and len(df) > 0 :
                            print(f"Warning: Validation set is empty after split. Val Ratio={self.val_ratio}, Total Rows={len(df)}, Remainder Rows={len(remaining_df)}. This might be due to small dataset size or stratification effects.")

                        # Add the 'subset' column back
                        train_df = train_df.copy(); train_df['subset'] = 'Train'
                        validation_df = validation_df.copy(); validation_df['subset'] = 'Validation'
                        test_df = test_df.copy(); test_df['subset'] = 'Test'

                        print(f"  [SourceStage] Random split final counts: Train={len(train_df)}, Val={len(validation_df)}, Test={len(test_df)}")

                    except ValueError as e_split:
                        print(f"\n!!! Error during scikit-learn train_test_split !!!")
                        print(f"Error: {e_split}")
                        print("This often happens with stratification when classes have too few samples (<2).")
                        print("Consider running again with stratify_column=None.")
                        raise RuntimeError(f"Random split failed: {e_split}")

                else: # Original logic: Split by existing 'subset' column
                    status_msg = "Source files created (column split)"
                    if 'subset' not in df.columns:
                        raise ValueError("Metadata CSV missing required 'subset' column for column-based split.")
                    print(f"  [SourceStage] Splitting based on existing 'subset' column.")
                    train_df = df[df['subset'].str.lower() == 'train'].copy() # Use .copy() to avoid SettingWithCopyWarning
                    test_df = df[df['subset'].str.lower() == 'test'].copy()
                    validation_df = df[df['subset'].str.lower().isin(['validation', 'val'])].copy()


                # --- Save the split DataFrames ---
                # Ensure dataframes exist before saving
                if train_df is not None: train_df.to_csv(train_csv, index=False)
                if test_df is not None: test_df.to_csv(test_csv, index=False)
                if validation_df is not None: validation_df.to_csv(validation_csv, index=False)

                if verbose and files_created:
                     print(f"  [SourceStage] Saved Train ({len(train_df if train_df is not None else [])} rows) to: {os.path.basename(train_csv)}")
                     print(f"  [SourceStage] Saved Test ({len(test_df if test_df is not None else [])} rows) to: {os.path.basename(test_csv)}")
                     print(f"  [SourceStage] Saved Validation ({len(validation_df if validation_df is not None else [])} rows) to: {os.path.basename(validation_csv)}")

            except FileNotFoundError:
                raise RuntimeError(f"Source metadata file not found at: {self.metadata_csv_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load or process metadata CSV from {self.metadata_csv_path}: {e}")

        # --- Print status and return ---
        if verbose:
            print("-------")
            print("Source stage")
            print(status_msg)
            print(f"Output Dir: {source_dir}")
            print("Version:", version_str)
            print("passed!")
            print("-------")

        # Return paths to the (now existing or previously cached) split files
        # Add checks to ensure files exist before returning paths if files_created is False? Or trust the initial check.
        # Assuming SourceData handles potential non-existence if needed downstream.
        return SourceData(train_csv, test_csv, validation_csv)
