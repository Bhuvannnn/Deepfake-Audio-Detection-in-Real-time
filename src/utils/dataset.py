import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_rate = 16000  # Fixed sample rate
        self.fixed_length = 32000  # Exactly 2 seconds (16000 * 2)
        
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
            
            # Ensure fixed length by either padding or trimming
            current_length = waveform.shape[1]
            
            if current_length > self.fixed_length:
                # Take the center part if too long
                start = (current_length - self.fixed_length) // 2
                waveform = waveform[:, start:start + self.fixed_length]
            else:
                # Pad with zeros if too short
                padding_length = self.fixed_length - current_length
                padding_left = padding_length // 2
                padding_right = padding_length - padding_left
                waveform = torch.nn.functional.pad(
                    waveform,
                    (padding_left, padding_right),
                    mode='constant',
                    value=0
                )
            
            # Verify the length
            assert waveform.shape[1] == self.fixed_length, \
                f"Waveform length {waveform.shape[1]} does not match fixed length {self.fixed_length}"
            
            # Normalize
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            return waveform.float()
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            # Return zeros with the correct fixed length
            return torch.zeros(1, self.fixed_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        try:
            waveform = self.load_audio(file_path)
            
            # Double-check the shape
            if waveform.shape[1] != self.fixed_length:
                print(f"Warning: Incorrect length {waveform.shape[1]} for {file_path}")
                waveform = torch.zeros(1, self.fixed_length, dtype=torch.float32)
            
            if self.transform:
                waveform = self.transform(waveform)
                
            return waveform, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros(1, self.fixed_length, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)