import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, duration: int):
        """
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories
            max_duration: Maximum duration in seconds for audio clips
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = 16000  # Fixed sample rate
        self.max_length = self.sample_rate * duration  # Convert seconds to samples
        
        # Get file paths
        self.real_files = list(Path(self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list(Path(self.data_dir / 'fake').glob('*.wav'))
        
        # Combine and create labels
        self.files = [(str(f), 0) for f in self.real_files] + \
                    [(str(f), 1) for f in self.fake_files]
    
    def load_audio(self, file_path: str):
        """Load and preprocess audio to ensure consistent length"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Handle length
            if waveform.shape[1] > self.max_length:
                # Take center section
                start = (waveform.shape[1] - self.max_length) // 2
                waveform = waveform[:, start:start + self.max_length]
            else:
                # Pad with zeros
                padding_length = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(
                    waveform, 
                    (0, padding_length),
                    mode='constant',
                    value=0
                )
            
            # Normalize
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            return waveform.float()
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return torch.zeros(1, self.max_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        waveform = self.load_audio(file_path)
        return waveform, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)