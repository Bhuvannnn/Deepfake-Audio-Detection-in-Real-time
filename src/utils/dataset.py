import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.processing.features import AudioPreprocessor
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, max_length: int = 16000*3):  # Reduced audio length
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
        # Get file paths
        self.real_files = list(Path(self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list(Path(self.data_dir / 'fake').glob('*.wav'))
        
        # Combine and create labels
        self.files = [(str(f), 0) for f in self.real_files] + \
                    [(str(f), 1) for f in self.fake_files]
    
    def load_audio(self, file_path: str):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or truncate to max_length
            if waveform.shape[1] < self.max_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_length - waveform.shape[1])
                )
            else:
                waveform = waveform[:, :self.max_length]
            
            # Convert to float32 and normalize
            waveform = waveform.to(torch.float32)
            waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return torch.zeros(1, self.max_length, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        waveform = self.load_audio(file_path)
        return waveform, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)