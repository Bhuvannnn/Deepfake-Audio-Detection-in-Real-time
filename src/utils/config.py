from pathlib import Path

class Config:
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    DURATION = 5  # seconds
    
    # Model parameters
    MODEL_NAME = "facebook/wav2vec2-base"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4