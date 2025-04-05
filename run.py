from Preprocess.Pipeline.Pipelines import SimplePipeline


if __name__ == "__main__":
    pipeline_version = "01"
    # Modified instantiation to call the class within the module.
    main_pipeline = SimplePipeline.SimplePipeline(pipeline_version=pipeline_version)
    train_loader, val_loader, test_loader = main_pipeline.run()

    # Demonstrate DataLoader content for the Train dataset.
    print("Final DataLoader for Train dataset. Sample contents:")
    for tensor_stack, label in train_loader:
        print("Tensor stack shape:", tensor_stack.shape)
        print("Label:", label)
        break
