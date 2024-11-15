import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_rate = 16000
        self.fixed_length = 32000  # 2 seconds
        
        # Get file paths
        self.real_files = list(Path(self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list(Path(self.data_dir / 'fake').glob('*.wav'))
        
        # Combine and create labels
        self.files = [(str(f), 0) for f in self.real_files] + \
                    [(str(f), 1) for f in self.fake_files]
        
        print(f"Found {len(self.real_files)} real and {len(self.fake_files)} fake audio files")
    
    def load_audio(self, file_path: str):
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
            
            # Ensure fixed length
            if waveform.shape[1] > self.fixed_length:
                start = (waveform.shape[1] - self.fixed_length) // 2
                waveform = waveform[:, start:start + self.fixed_length]
            else:
                padding_length = self.fixed_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, padding_length),
                    mode='constant',
                    value=0
                )
            
            # Normalize
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Ensure shape is [fixed_length]
            waveform = waveform.squeeze()  # Remove any extra dimensions
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return torch.zeros(self.fixed_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        try:
            waveform = self.load_audio(file_path)
            
            if self.transform:
                waveform = self.transform(waveform)
            
            # Verify shape is correct
            if waveform.shape[0] != self.fixed_length:
                print(f"Warning: Incorrect shape {waveform.shape} for {file_path}")
                waveform = torch.zeros(self.fixed_length, dtype=torch.float32)
                
            return waveform, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros(self.fixed_length, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)