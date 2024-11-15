import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, duration: int = 5):
        self.sample_rate = sample_rate
        self.duration = duration
        
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file."""
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            return waveform, self.sample_rate
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def extract_features(self, waveform: torch.Tensor) -> dict:
        """Extract various audio features."""
        # Convert to numpy for librosa compatibility
        audio_np = waveform.numpy().squeeze()
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=self.sample_rate)
        
        return {
            'mfcc': torch.from_numpy(mfcc),
            'spectral_centroid': torch.from_numpy(spectral_centroid),
            'spectral_rolloff': torch.from_numpy(spectral_rolloff)
        }