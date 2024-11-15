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
            
            # Ensure shape is [fixed_length] by removing all extra dimensions
            waveform = waveform.squeeze()
            
            # If squeeze removed too many dimensions, add one back
            if waveform.dim() == 1:
                return waveform  # Return as [fixed_length]
            else:
                return waveform.reshape(self.fixed_length)  # Force correct shape
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return torch.zeros(self.fixed_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        try:
            waveform = self.load_audio(file_path)
            
            if self.transform:
                waveform = self.transform(waveform)
            
            # Double check the shape is correct
            if waveform.shape != torch.Size([self.fixed_length]):
                print(f"Fixing shape from {waveform.shape} to [{self.fixed_length}]")
                waveform = waveform.reshape(self.fixed_length)
                
            return waveform, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros(self.fixed_length, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)

def verify_dataset_shapes(dataset):
    """Verify the shapes of dataset samples"""
    sample_waveform, sample_label = dataset[0]
    print(f"Sample waveform shape: {sample_waveform.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Fixed length should be: {dataset.fixed_length}")
    
    # Check if the shape is exactly [fixed_length]
    expected_shape = torch.Size([dataset.fixed_length])
    is_correct = sample_waveform.shape == expected_shape
    
    if not is_correct:
        print(f"ERROR: Expected shape {expected_shape}, got {sample_waveform.shape}")
    else:
        print("Shape verification passed!")
    
    return is_correct

# In main execution: