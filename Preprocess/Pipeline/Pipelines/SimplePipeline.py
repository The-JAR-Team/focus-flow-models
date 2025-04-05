from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
from Preprocess.Pipeline.Stages.SourceStage import SourceStage


class MainPipeline:
    """
    Main pipeline that orchestrates the entire preprocessing flow:
      1. Run the source stage to obtain the CSVs.
      2. For each row in the source CSV (e.g., Train CSV), run the inner pipeline.
      3. Run a final stage that takes all inner pipeline outputs and produces a DataLoader.
    """
    def __init__(self, source_stage, inner_pipeline, final_stage):
        self.source_stage = source_stage      # Instance of SourceStage
        self.inner_pipeline = inner_pipeline  # An OrchestrationPipeline instance for inner processing
        self.final_stage = final_stage        # A final stage instance to produce the DataLoader

    def run(self):
        # Step 1: Run the source stage to get SourceData.
        source_data = self.source_stage.process()

        # For this example, we'll use the Train CSV.
        print("Loading Train CSV from source data...")
        train_df = source_data.get_train_data()

        inner_outputs = []
        # Step 2: For each row in the Train CSV, run the inner pipeline.
        # for idx, row in train_df.iterrows():
            # print(f"Processing row {idx} from Train CSV...")
            # Each row is passed as a pandas Series.
            # result = self.inner_pipeline.run(row)
            # inner_outputs.append(result)

        # Step 3: Run the final stage to create a DataLoader.
        dataloader = self.final_stage.process(inner_outputs)
        return dataloader


class DummyInnerStage:
    """
    A dummy stage for the inner pipeline.
    In your real implementation, this would perform processing on a video clip or frame.
    """
    def process(self, data):
        # For example, print and mark the row as processed.
        print(f"Processing inner pipeline for clip_folder: {data['clip_folder']}")
        processed_data = data.to_dict()
        processed_data['processed'] = True
        return processed_data


class FinalStage:
    """
    A dummy final stage that takes all inner outputs and creates a DataLoader.
    Replace this with your actual DataLoader creation logic.
    """
    def process(self, data):
        print("Final stage: creating DataLoader from processed data...")
        # Here we create a dummy DataLoader that simply wraps the list of processed items.
        class DummyDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        return DummyDataLoader(data)


if __name__ == "__main__":
    # Set up the SourceStage with the pipeline version (e.g., "01")
    pipeline_version = "01"
    source_stage = SourceStage(pipeline_version)

    # Set up the inner pipeline.
    # Here we create a simple OrchestrationPipeline with a single dummy inner stage.
    inner_pipeline = OrchestrationPipeline(stages=[DummyInnerStage()])

    # Set up the final stage.
    final_stage = FinalStage()

    # Create the MainPipeline with the three parts.
    main_pipeline = MainPipeline(source_stage, inner_pipeline, final_stage)

    # Run the MainPipeline.
    dataloader = main_pipeline.run()

    # Demonstrate the DataLoader by iterating over it.
    print("Final DataLoader created. Contents:")
    for item in dataloader:
        print(item)