from build import ModelPipeline

# Initialize pipeline
pipeline = ModelPipeline(target_col="liquidity_stress_next_30d")

# Run automated split, feature extraction, model fitting, and validation evaluation
pipeline.fit_and_evaluate("Train.csv", val_size=0.2, version=3, threshold=0.5)
