import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Custom layers for artifact detection
        self.artifact_detector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )
        
    def forward(self, x):
        # Get Wav2Vec features
        features = self.wav2vec(x).last_hidden_state
        
        # Pool features (mean pooling)
        pooled = torch.mean(features, dim=1)
        
        # Artifact detection
        output = self.artifact_detector(pooled)
        
        return output

    @classmethod
    def load_from_disk(cls, path: str):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model