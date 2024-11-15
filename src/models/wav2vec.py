import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained Wav2Vec model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze wav2vec parameters (optional)
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        # Custom layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # x should be [batch_size, sequence_length]
        # Print shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Wav2Vec expects input shape [batch_size, sequence_length]
        if x.dim() == 3:  # [batch_size, 1, sequence_length]
            x = x.squeeze(1)
        elif x.dim() == 1:  # Single sample
            x = x.unsqueeze(0)
            
        print(f"Shape after processing: {x.shape}")
        
        # Get Wav2Vec features
        outputs = self.wav2vec(x)
        features = outputs.last_hidden_state
        
        # Pool features (mean pooling)
        features = torch.mean(features, dim=1)
        
        # Classification
        output = self.classifier(features)
        
        return output