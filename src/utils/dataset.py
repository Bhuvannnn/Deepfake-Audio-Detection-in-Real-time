import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 16000, duration: int = 3):
        """
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories
            sample_rate: Audio sample rate
            duration: Duration in seconds for all audio clips
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration
        
        # Get file paths
        self.real_files = list(Path(self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list(Path(self.data_dir / 'fake').glob('*.wav'))
        
        # Combine and create labels
        self.files = [(str(f), 0) for f in self.real_files] + \
                    [(str(f), 1) for f in self.fake_files]
        
        # Initialize resampler
        self.resampler = None
    
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
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=sr,
                        new_freq=self.sample_rate
                    )
                waveform = self.resampler(waveform)
            
            # Handle length
            if waveform.shape[1] > self.target_length:
                # Take center section
                start = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            else:
                # Pad with zeros
                padding_length = self.target_length - waveform.shape[1]
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
            return torch.zeros(1, self.target_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        file_path, label = self.files[idx]
        waveform = self.load_audio(file_path)
        return waveform, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.files)