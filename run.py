from Preprocess.Pipeline.Pipelines.ConfigurablePipeline import load_pipeline_from_json

if __name__ == "__main__":
    # Example 1: Load from a JSON file
    config_path = "Preprocess/Pipeline/Pipelines/configs/02_10fps_quality95.json"
    pipeline = load_pipeline_from_json(config_path)
    train_loader, val_loader, test_loader = pipeline.run()

    # Demonstrate DataLoader content for the Train dataset
    print("Final DataLoader for Train dataset. Sample contents:")
    for tensor_stack, label in train_loader:
        print("Tensor stack shape:", tensor_stack.shape)
        print("Label:", label)
        break
