import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.processing.features import AudioPreprocessor

class AudioDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the dataset
        Args:
            data_dir (str): Directory containing 'real' and 'fake' subdirectories
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.preprocessor = AudioPreprocessor()
        
        # Get all audio files
        self.real_files = list(Path(self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list(Path(self.data_dir / 'fake').glob('*.wav'))
        
        # Combine real and fake files with labels
        self.files = [(str(f), 0) for f in self.real_files] + [(str(f), 1) for f in self.fake_files]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        """
        file_path, label = self.files[idx]
        
        # Load and preprocess audio
        try:
            waveform, _ = self.preprocessor.load_audio(file_path)
            
            # Convert to float and normalize if needed
            if self.transform:
                waveform = self.transform(waveform)
            
            return waveform, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Return a zero tensor with correct shape in case of error
            return torch.zeros(1, 16000), torch.tensor(0, dtype=torch.long)